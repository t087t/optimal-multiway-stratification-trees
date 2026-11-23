"""CART leaf stratification and population-mean RMSE evaluation."""

from typing import Optional

import graphviz
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor, export_graphviz, export_text
from sklearn.utils.validation import check_is_fitted, check_X_y

from src.allocator import AllocatorProtocol, estimate_y_rmse


class CARTEstimator(RegressorMixin, BaseEstimator):  # type: ignore
    """Learn CART leaf strata and evaluate a stratified population-mean estimator.

    Attributes:
        model_: Fitted decision-tree regressor.
        leaf_id_map_: Mapping from tree leaf IDs to consecutive stratum labels.
        sample_size_each_cluster: Integer allocation learned from training leaf strata.
    """

    def __init__(
        self,
        allocator: AllocatorProtocol,
        sample_size: int,
        max_leaf_nodes: Optional[int] = None,
        max_depth: Optional[int] = None,
        n_trials: int = 100,
        random_state: int = 0,
        min_samples_leaf: int = 1,
    ):
        """Store estimator settings without fitting a model.

        Args:
            allocator: Allocator used to determine integer sample counts for each stratum.
            sample_size: Total number of observations to sample without replacement.
            max_leaf_nodes: Maximum decision-tree leaves. None leaves this limit unset.
            max_depth: Maximum decision-tree depth. None leaves this limit unset.
            n_trials: Number of Monte Carlo sampling trials.
            random_state: Initial random seed for model fitting and Monte Carlo sampling.
            min_samples_leaf: Minimum training observations in each CART leaf.

        Returns:
            None: Settings are stored for subsequent fitting and evaluation.
        """
        self.allocator = allocator
        self.sample_size = sample_size
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.n_trials = n_trials
        self.random_state = random_state
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "CARTEstimator":
        """Fit a CART tree and learn the sample allocation for its leaves.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Population outcomes of shape (n_samples,), aligned with the feature rows.

        Returns:
            CARTEstimator: This fitted estimator.

        Raises:
            ValueError: If training inputs or model parameters are invalid.
            AssertionError: If the configured allocator rejects its resulting sample counts.
        """
        X_df_or_array = X
        X, y = check_X_y(X, y, y_numeric=True)

        self.model_ = DecisionTreeRegressor(
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            random_state=self.random_state,
            min_samples_leaf=self.min_samples_leaf,
        )
        self.model_.fit(X, y)

        labels_from_tree = self.model_.apply(X)

        # Allocation helpers require consecutive labels rather than tree node IDs.
        unique_leaf_ids = np.unique(labels_from_tree)
        self.n_leaves_ = len(unique_leaf_ids)
        self.leaf_id_map_ = {leaf_id: i for i, leaf_id in enumerate(unique_leaf_ids)}
        labels = np.array([self.leaf_id_map_[label] for label in labels_from_tree])

        self.sample_size_each_cluster = self.allocator.solve(
            y=y,
            labels=labels,
            n_clusters=self.n_leaves_,
            sample_size=self.sample_size,
        )

        self.is_fitted_ = True

        try:
            feature_names = X_df_or_array.columns.tolist()
        except AttributeError:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        tree_rules = export_text(self.model_, feature_names=feature_names)
        print("--- CART Stratification Rules ---")
        print(tree_rules)
        print("---------------------------------")

        return self

    def predict(self, X: NDArray[np.float64], y: NDArray[np.float64], true_mean: float) -> float:
        """Estimate population-mean RMSE over repeated sampling trials.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
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
        check_is_fitted(self, "is_fitted_")
        X, y = check_X_y(X, y, y_numeric=True)

        leaf_ids_pred = self.model_.apply(X)
        labels = np.array([self.leaf_id_map_.get(label, -1) for label in leaf_ids_pred])

        return estimate_y_rmse(
            sample_size_each_cluster=self.sample_size_each_cluster,
            y=y,
            labels=labels,
            true_mean=true_mean,
            n_trials=self.n_trials,
            random_state=self.random_state,
        )

    def save_tree_png(self, feature_names: list, filename: str = "cart_tree.png"):
        """Render the fitted CART tree to a PNG file.

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
        check_is_fitted(self)

        font = "Yu Gothic UI Bold"

        dot_data = export_graphviz(
            self.model_,
            out_file=None,
            feature_names=feature_names,
            filled=True,
            rounded=True,
            special_characters=True,
            fontname=font,
        )

        dot_data = dot_data.replace("samples =", "事例数: ")
        dot_data = dot_data.replace("value =", "予測値: ")
        dot_data = dot_data.replace("squared_error =", "平均二乗誤差: ")

        settings = (
            f'graph [dpi=300, labelfontname="{font}", labelfontsize=10, margin=0.2];\n'
            f'    node [fontname="{font}", fontsize=9];'
        )
        dot_data = dot_data.replace("graph [", settings + "\n    graph [", 1)

        graph = graphviz.Source(dot_data)

        graph.render(filename.rsplit(".", 1)[0], format="png", cleanup=True)
        print(f"決定木を '{filename}' に保存しました。")

    def __str__(self) -> str:
        """Return the method label used in experiment tables.

        Returns:
            str: Human-readable method name.
        """
        return "CART"
