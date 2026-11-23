"""KMeans and Gaussian-mixture stratification."""

from typing import Union

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from src.allocator import AllocatorProtocol, estimate_y_rmse

CLUSTERRING_CLASS = Union[KMeans, GaussianMixture]


class NormalEstimator(RegressorMixin, BaseEstimator):  # type: ignore
    """Learn KMeans or Gaussian-mixture strata and their sample allocation.

    Attributes:
        model: Fitted clustering model.
        sample_size_each_cluster: Integer allocation learned from training clusters.
    """

    def __init__(
        self,
        n_clusters: int,
        allocator: AllocatorProtocol,
        sample_size: int,
        n_trials: int = 100,
        clustering_method: str = "KMeans",
        random_state: int = 0,
    ):
        """Store estimator settings without fitting a model.

        Args:
            n_clusters: Number of strata or clusters.
            allocator: Allocator used to determine integer sample counts for each stratum.
            sample_size: Total number of observations to sample without replacement.
            n_trials: Number of Monte Carlo sampling trials.
            clustering_method: Clustering method: "KMeans" or "GMM".
            random_state: Initial random seed for model fitting and Monte Carlo sampling.

        Returns:
            None: Settings are stored for subsequent fitting and evaluation.
        """
        self.n_clusters = n_clusters
        self.allocator = allocator
        self.n_trials = n_trials
        self.sample_size = sample_size
        self.clustering_method = clustering_method
        self.random_state = random_state

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "NormalEstimator":
        """Fit the clustering model and learn a stratum allocation.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Population outcomes of shape (n_samples,), aligned with the feature rows.

        Returns:
            NormalEstimator: This fitted estimator.

        Raises:
            ValueError: If training inputs or model parameters are invalid.
            AttributeError: If clustering_method is unsupported and no clustering model is
                available.
            AssertionError: If the configured allocator rejects its resulting sample counts.
        """
        X, y = check_X_y(X, y)

        self.model: CLUSTERRING_CLASS
        if self.clustering_method == "GMM":
            self.model = GaussianMixture(
                n_components=self.n_clusters,
                random_state=self.random_state,
                init_params="kmeans",
            )

        if self.clustering_method == "KMeans":
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
            )
        self.model.fit(X)
        labels = self.model.predict(X)
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

        labels = self.model.predict(X)

        return estimate_y_rmse(
            sample_size_each_cluster=self.sample_size_each_cluster,
            y=y,
            labels=labels,
            true_mean=true_mean,
            n_trials=self.n_trials,
            random_state=self.random_state,
        )

    def __str__(self) -> str:
        """Return the method label used in experiment tables.

        Returns:
            str: Human-readable method name.
        """
        return self.clustering_method
