"""KMeans-guided decision-tree leaf stratification."""

import re

import graphviz
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from src.allocator import AllocatorProtocol, estimate_y_rmse


class KMeansDTEstimator(RegressorMixin, BaseEstimator):  # type: ignore
    """Learn tree-leaf strata from KMeans labels and allocate samples by leaf.

    Attributes:
        kmeans_: Fitted clustering model on standardized training features.
        dt_model_: Tree fitted to predict KMeans labels from unstandardized features.
        sample_size_each_stratum_: Integer allocation learned for the tree leaves.
        majority_cluster_each_stratum_: Representative KMeans label for each leaf.
    """

    def __init__(
        self,
        n_clusters: int,
        allocator: AllocatorProtocol,
        sample_size: int,
        n_trials: int = 100,
        random_state: int = 0,
        max_leaf_nodes: int = 8,
    ):
        """Store estimator settings without fitting a model.

        Args:
            n_clusters: Number of strata or clusters.
            allocator: Allocator used to determine integer sample counts for each stratum.
            sample_size: Total number of observations to sample without replacement.
            n_trials: Number of Monte Carlo sampling trials.
            random_state: Initial random seed for model fitting and Monte Carlo sampling.
            max_leaf_nodes: Maximum decision-tree leaves. None leaves this limit unset.

        Returns:
            None: Settings are stored for subsequent fitting and evaluation.
        """
        self.n_clusters = n_clusters
        self.allocator = allocator
        self.n_trials = n_trials
        self.sample_size = sample_size
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes

    def fit(
        self,
        X_std: NDArray[np.float64],
        y: NDArray[np.float64],
        X_orig: NDArray[np.float64],
    ) -> "KMeansDTEstimator":
        """Fit KMeans, a cluster-prediction tree, and leaf-level allocation.

        Args:
            X_std: Standardized features for KMeans, with shape (n_samples, n_features).
            y: Population outcomes of shape (n_samples,), aligned with the feature rows.
            X_orig: Unstandardized encoded features for the decision tree, aligned with X_std.

        Returns:
            KMeansDTEstimator: This fitted estimator.

        Raises:
            ValueError: If training inputs or model parameters are invalid.
            AssertionError: If the configured allocator rejects its resulting sample counts.
        """
        X_std, y = check_X_y(X_std, y, dtype=np.float64)
        X_orig = check_array(X_orig, dtype=np.float64)

        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init="auto",
        )
        kmeans_labels: NDArray[np.int_] = self.kmeans_.fit_predict(X_std).astype(int)

        self.dt_model_ = DecisionTreeClassifier(
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
        )
        self.dt_model_.fit(X_orig, kmeans_labels)

        leaf_node_ids: NDArray[np.int_] = self.dt_model_.apply(X_orig).astype(int)

        # Use consecutive leaf labels because allocation is performed by stratum.
        unique_leaf_ids = np.unique(leaf_node_ids)
        self.leaf_id_to_stratum_ = {leaf_id: i for i, leaf_id in enumerate(unique_leaf_ids)}
        self.stratum_to_leaf_id_ = {i: leaf_id for leaf_id, i in self.leaf_id_to_stratum_.items()}

        strata_labels = np.vectorize(self.leaf_id_to_stratum_.get)(leaf_node_ids).astype(int)
        self.n_strata_ = int(unique_leaf_ids.size)

        self.sample_size_each_stratum_ = self.allocator.solve(
            y=y,
            labels=strata_labels,
            n_clusters=self.n_strata_,
            sample_size=self.sample_size,
        )

        # Majority cluster labels are for interpretation; leaves define the strata.
        majority = np.zeros(self.n_strata_, dtype=int)
        for h in range(self.n_strata_):
            km_in_h = kmeans_labels[strata_labels == h]
            if km_in_h.size == 0:
                majority[h] = -1
            else:
                counts = np.bincount(km_in_h, minlength=self.n_clusters)
                majority[h] = int(np.argmax(counts))
        self.majority_cluster_each_stratum_ = majority

        self.is_fitted_ = True
        return self

    def predict_strata(self, X_orig: NDArray[np.float64]) -> NDArray[np.int_]:
        """Map evaluation rows to consecutive decision-tree leaf labels.

        Args:
            X_orig: Unstandardized encoded features in the fitted tree column order.

        Returns:
            ndarray: Stratum labels from zero to n_strata_ - 1.

        Raises:
            NotFittedError: If the estimator has not been fitted.
            ValueError: If features are invalid or a leaf is absent from the fitted label map.
        """
        check_is_fitted(self, attributes=["dt_model_", "leaf_id_to_stratum_", "n_strata_"])
        X_orig = check_array(X_orig, dtype=np.float64)

        leaf_node_ids = self.dt_model_.apply(X_orig).astype(int)

        try:
            strata_labels = np.vectorize(self.leaf_id_to_stratum_.__getitem__)(
                leaf_node_ids
            ).astype(int)
        except KeyError as e:
            raise ValueError(
                "未知のleaf_idが出ました。通常は起こりませんが、モデル破損/不整合の可能性があります。"
            ) from e
        return strata_labels

    def predict_cluster(self, X_orig: NDArray[np.float64]) -> NDArray[np.int_]:
        """Return each leaf stratum's majority KMeans cluster.

        Args:
            X_orig: Unstandardized encoded features in the fitted tree column order.

        Returns:
            ndarray: Representative cluster label for each evaluation row.

        Raises:
            NotFittedError: If the estimator has not been fitted.
            ValueError: If the evaluation features or leaf mapping are invalid.
        """
        check_is_fitted(self, attributes=["majority_cluster_each_stratum_"])
        strata = self.predict_strata(X_orig)
        return self.majority_cluster_each_stratum_[strata]

    def predict(
        self,
        X_orig: NDArray[np.float64],
        y: NDArray[np.float64],
        true_mean: float,
    ) -> float:
        """Estimate population-mean RMSE over repeated sampling trials.

        Args:
            X_orig: Unstandardized encoded evaluation features in the fitted tree column order.
            y: Population outcomes of shape (n_samples,), aligned with the feature rows.
            true_mean: Mean of the full population against which estimates are evaluated.

        Returns:
            float: Monte Carlo RMSE, not a vector of row-level predictions.

        Raises:
            NotFittedError: If the estimator has not been fitted.
            ValueError: If inputs have invalid shapes or values, or an allocated draw exceeds its
                stratum population.

        Notes:
            Uses the allocation learned during fit; it does not reallocate on evaluation data.
        """
        check_is_fitted(self, attributes=["sample_size_each_stratum_"])
        X_orig = check_array(X_orig, dtype=np.float64)
        y = check_array(y, dtype=np.float64, ensure_2d=False)

        labels = self.predict_strata(X_orig)

        return estimate_y_rmse(
            sample_size_each_cluster=self.sample_size_each_stratum_,
            y=y,
            labels=labels,
            true_mean=true_mean,
            n_trials=self.n_trials,
            random_state=self.random_state,
        )

    @staticmethod
    def lighten_color(hex_color, factor=0.4):
        """Mix an RGB hexadecimal color toward white.

        Args:
            hex_color (str): RGB color as six hexadecimal digits, with an optional leading "#".
            factor (float): Fraction mixed toward white, conventionally between 0 and 1.

        Returns:
            str: Blended color in #rrggbb form.

        Raises:
            ValueError: If a color component cannot be parsed as hexadecimal or converted to an
                integer.
        """
        hex_color = hex_color.lstrip("#")

        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)

        return f"#{r:02x}{g:02x}{b:02x}"

    def save_decision_tree(self, feature_names: list, filename: str = "decision_tree.png") -> None:
        """Render the fitted cluster-prediction tree to PNG.

        Args:
            feature_names: Feature names in the order used to fit the decision tree.
            filename: Output PNG path; the renderer manages the extension and temporary DOT file.

        Returns:
            None: The rendered tree is written to filename.

        Raises:
            NotFittedError: If the estimator has not been fitted.
            graphviz.ExecutableNotFound: If the Graphviz dot executable is unavailable.
            subprocess.CalledProcessError: If Graphviz cannot render the graph.
            OSError: If the output files cannot be written.
            ValueError: If feature_names does not match the fitted tree.
        """
        check_is_fitted(self, attributes=["dt_model_"])

        font = "Yu Gothic UI Bold"

        dot_data = export_graphviz(
            self.dt_model_,
            out_file=None,
            feature_names=feature_names,
            class_names=[f"クラスタ {i}" for i in range(self.n_clusters)],
            filled=True,
            rounded=True,
            special_characters=True,
            fontname=font,
        )

        dot_data = dot_data.replace("samples =", "事例数: ")
        dot_data = dot_data.replace("value =", "クラス別事例数: ")
        dot_data = dot_data.replace("gini =", "ジニ係数: ")
        dot_data = dot_data.replace("class =", "予測クラス: ")

        settings = (
            f'graph [dpi=300, labelfontname="{font}", labelfontsize=10, margin=0.2];\n'
            f'    node [fontname="{font}", fontsize=12, margin="0.3,0.15"];'
        )
        dot_data = dot_data.replace("graph [", settings + "\n    graph [", 1)

        lighten_factor = 0.2
        dot_data = re.sub(
            r'fillcolor="(#[A-Fa-f0-9]{6})"',
            lambda m: f'fillcolor="{self.lighten_color(m.group(1), factor=lighten_factor)}"',
            dot_data,
        )

        graph = graphviz.Source(dot_data)
        graph.render(filename.replace(".png", ""), format="png", cleanup=True)
        print(f"決定木を '{filename}' に保存しました。")

    def __str__(self) -> str:
        """Return the method label used in experiment tables.

        Returns:
            str: Human-readable method name.
        """
        return "KMeans-DT"
