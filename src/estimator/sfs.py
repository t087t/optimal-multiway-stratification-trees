"""Sequential forward selection of clustering features."""

import warnings
from typing import Union

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from src.allocator import AllocatorProtocol, compute_y_var, estimate_y_rmse

CLUSTERRING_CLASS = Union[KMeans, GaussianMixture]


class BaseSequentialForwardSelectionEstimator(RegressorMixin, BaseEstimator):  # type: ignore
    """Select clustering features greedily using a subclass-defined score.

    Attributes:
        selected_features_index: Input-column indices in feature-selection order.
        score_history: Score after each accepted feature addition.
        model: Clustering model fitted on the selected features.
        sample_size_each_cluster: Integer allocation learned from training data.
    """

    def __init__(
        self,
        maximum_features_to_select: int,
        n_clusters: int,
        allocator: AllocatorProtocol,
        sample_size: int,
        n_trials: int = 100,
        clustering_method: str = "KMeans",
        select_maximum_features: bool = True,
        random_state: int = 0,
    ):
        """Store estimator settings without fitting a model.

        Args:
            maximum_features_to_select: Upper bound on the number of selected features.
            n_clusters: Number of strata or clusters.
            allocator: Allocator used to determine integer sample counts for each stratum.
            sample_size: Total number of observations to sample without replacement.
            n_trials: Number of Monte Carlo sampling trials.
            clustering_method: Clustering method: "KMeans" or "GMM".
            select_maximum_features: Whether to keep selecting until the feature limit; otherwise
                stop when the score does not improve.
            random_state: Initial random seed for model fitting and Monte Carlo sampling.

        Returns:
            None: Settings are stored for subsequent fitting and evaluation.
        """
        self.maximum_features_to_select = maximum_features_to_select
        self.allocator = allocator
        self.sample_size = sample_size
        self.n_trials = n_trials
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.select_maximum_features = select_maximum_features
        self.random_state = random_state

    def fit(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> "BaseSequentialForwardSelectionEstimator":
        """Select features greedily and learn a cluster allocation.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Population outcomes of shape (n_samples,), aligned with the feature rows.

        Returns:
            BaseSequentialForwardSelectionEstimator: This fitted estimator.

        Raises:
            ValueError: If training inputs or model parameters are invalid.
            NotImplementedError: If the base scoring method is used without an implementation.
            UnboundLocalError: If no candidate model is selected, for example with a nonpositive
                feature limit.
            AssertionError: If the configured allocator rejects its resulting sample counts.
        """
        X, y = check_X_y(X, y)
        n_all_features = X.shape[1]

        self.score_history: list[float] = []

        current_features: list[int] = []
        remaining_features = list(range(n_all_features))

        if not self.select_maximum_features:
            best_score = -np.inf

        while len(current_features) < self.maximum_features_to_select:
            best_feature = None

            if self.select_maximum_features:
                best_score = -np.inf

            for feature in remaining_features:
                temp_features = current_features + [feature]
                score, model = self.compute_score(X[:, temp_features], y)

                # Strict improvement retains the first feature when candidate scores tie.
                if best_score < score:
                    best_score = score
                    best_feature = feature
                    best_model = model

            if best_feature is not None:
                current_features.append(best_feature)
                remaining_features.remove(best_feature)
                self.score_history.append(best_score)
            else:
                break

        self.selected_features_index = current_features
        self.model = best_model

        labels = self.model.predict(X[:, self.get_selected_features_index()])
        self.sample_size_each_cluster = self.allocator.solve(
            y, labels, self.n_clusters, self.sample_size
        )

        self.is_fitted_ = True

        return self

    def predict(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        true_mean: float,
    ) -> float:
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
        X = check_array(X, dtype=np.float64)
        y = check_array(y, dtype=np.float64, ensure_2d=False)

        labels = self.model.predict(X[:, self.get_selected_features_index()])

        return estimate_y_rmse(
            sample_size_each_cluster=self.sample_size_each_cluster,
            y=y,
            labels=labels,
            true_mean=true_mean,
            n_trials=self.n_trials,
            random_state=self.random_state,
        )

    def compute_score(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> tuple[float, CLUSTERRING_CLASS]:
        """Define the scoring operation required by feature-selection subclasses.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Population outcomes of shape (n_samples,), aligned with the feature rows.

        Returns:
            NoReturn: The base implementation always raises NotImplementedError.

        Raises:
            NotImplementedError: Always; subclasses must implement a scoring rule.
        """
        raise NotImplementedError

    def clustering(self, X: NDArray[np.float64]) -> tuple[CLUSTERRING_CLASS, NDArray[np.int_]]:
        """Fit the configured clustering model on a candidate feature subset.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            tuple: Fitted clustering model and its training-row labels.

        Raises:
            ValueError: If the method is not KMeans/GMM or the clustering input is invalid.
        """
        if self.clustering_method == "GMM":
            model = GaussianMixture(
                n_components=self.n_clusters,
                random_state=self.random_state,
                init_params="kmeans",
            )

        elif self.clustering_method == "KMeans":
            model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
            )
        else:
            raise ValueError("Invalid clustering method. Please select 'GMM' or 'KMeans'.")
        model.fit(X)
        labels = model.predict(X)

        return model, labels

    def get_selected_features_index(self) -> list[int]:
        """Return selected feature indices in selection order.

        Returns:
            list[int]: Indices of selected input columns.

        Raises:
            AttributeError: If feature selection has not been fitted.
        """
        return self.selected_features_index

    def get_score_history(self) -> list[float]:
        """Return the score after each accepted feature addition.

        Returns:
            list[float]: Feature-selection scores in acceptance order.

        Raises:
            AttributeError: If feature selection has not been fitted.
        """
        return self.score_history


class SequentialForwardSelectionVarEstimator(BaseSequentialForwardSelectionEstimator):
    """Select features by minimizing analytic stratified-mean variance."""

    def compute_score(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> tuple[float, BaseEstimator]:
        """Score a candidate subset by negative stratified-mean variance.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Population outcomes of shape (n_samples,), aligned with the feature rows.

        Returns:
            tuple: Negative analytic variance and the fitted clustering model.

        Raises:
            ValueError: If clustering inputs or allocation labels are invalid.
            AssertionError: If the allocator rejects its resulting sample counts.
        """
        model, labels = self.clustering(X)
        sample_size_each_cluster = self.allocator.solve(
            y, labels, self.n_clusters, self.sample_size
        )
        var = compute_y_var(sample_size_each_cluster, y, labels)

        # Selection maximizes scores, so negate the variance objective.
        score = -var

        return score, model

    def __str__(self) -> str:
        """Return the method label used in experiment tables.

        Returns:
            str: Human-readable method name.
        """
        return f"SFS-{self.clustering_method}-Var"


class SequentialForwardSelectionFEstimator(BaseSequentialForwardSelectionEstimator):
    """Select features by maximizing the clustering model's own score."""

    def compute_score(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> tuple[float, BaseEstimator]:
        """Score a candidate subset using the clustering model's own score.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Outcomes accepted for the scoring interface but not used.

        Returns:
            tuple: Model score and fitted clustering model.

        Raises:
            ValueError: If the clustering method or input is invalid.

        Notes:
            KMeans uses negative inertia; GMM uses average log-likelihood. The outcome y is not
            used.
        """
        model, _ = self.clustering(X)
        score = model.score(X)

        return score, model

    def __str__(self) -> str:
        """Return the method label used in experiment tables.

        Returns:
            str: Human-readable method name.
        """
        return f"SFS-{self.clustering_method}-F"


class SequentialForwardSelectionTEstimator(BaseSequentialForwardSelectionEstimator):
    """Select features by minimizing within-cluster outcome sums of squares."""

    def compute_score(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> tuple[float, BaseEstimator]:
        """Score a candidate subset by negative within-cluster outcome sums of squares.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Population outcomes of shape (n_samples,), aligned with the feature rows.

        Returns:
            tuple: Negative outcome sum of squares and the fitted clustering model.

        Raises:
            ValueError: If the clustering method or input is invalid.

        Notes:
            Non-KMeans clustering emits the existing warning and still computes the score.
        """
        model, labels = self.clustering(X)
        if self.clustering_method != "KMeans":
            warnings.warn(
                "Total Sum of Squares is only available for KMeans clustering. if you use other clustering methods, Unexpected result may be returned."
            )

        tss = 0
        for k in np.unique(labels):
            y_k = y[labels == k]
            mean_y_k = np.mean(y_k)
            tss += np.sum((y_k - mean_y_k) ** 2)
        score = -tss

        return score, model

    def __str__(self) -> str:
        """Return the method label used in experiment tables.

        Returns:
            str: Human-readable method name.
        """
        return f"SFS-{self.clustering_method}-T"
