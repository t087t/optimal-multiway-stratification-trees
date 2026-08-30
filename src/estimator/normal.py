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
    def __init__(
        self,
        n_clusters: int,
        allocator: AllocatorProtocol,
        sample_size: int,
        n_trials: int = 100,
        clustering_method: str = "KMeans",
        random_state: int = 0,
    ):
        self.n_clusters = n_clusters
        self.allocator = allocator
        self.n_trials = n_trials
        self.sample_size = sample_size
        self.clustering_method = clustering_method
        self.random_state = random_state

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "NormalEstimator":
        """クラスタリングを実行"""
        X, y = check_X_y(X, y)

        self.model: CLUSTERRING_CLASS
        if self.clustering_method == "GMM":
            self.model = GaussianMixture(
                n_components=self.n_clusters,
                random_state=self.random_state,
                init_params="kmeans",
            )
        # クラスタリング手法がKMEANSの場合
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
        true_mean: float,  # true_mean引数を追加
    ) -> float:
        """処置群の標本RMSEの推定"""
        check_is_fitted(self, "is_fitted_")
        X = check_array(X, dtype=np.float64)
        y = check_array(y, dtype=np.float64, ensure_2d=False)

        # 層化
        labels = self.model.predict(X)

        # estimate_y_rmse を呼び出す
        return estimate_y_rmse(
            sample_size_each_cluster=self.sample_size_each_cluster,
            y=y,
            labels=labels,
            true_mean=true_mean,
            n_trials=self.n_trials,
            random_state=self.random_state,
        )

    def __str__(self) -> str:
        return self.clustering_method
