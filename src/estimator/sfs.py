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
        """特徴量選択とクラスタリングを実行"""
        X, y = check_X_y(X, y)
        n_all_features = X.shape[1]  # 総特徴量数

        self.score_history: list[float] = []

        # 選ばれた特徴量と残っている特徴量の初期化
        current_features: list[int] = []
        remaining_features = list(range(n_all_features))

        if not self.select_maximum_features:
            best_score = -np.inf

        while len(current_features) < self.maximum_features_to_select:
            best_feature = None  # 選ぶ特徴量の初期化

            if self.select_maximum_features:
                best_score = -np.inf

            # 未選択の特徴量の中から最もスコアが高い特徴量を選択する
            for feature in remaining_features:
                temp_features = current_features + [feature]  # 特徴量をひとつ加え、score計算
                score, model = self.compute_score(X[:, temp_features], y)

                # スコアが最も高い特徴量を選択
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
        true_mean: float,  # true_mean引数を追加
    ) -> float:
        """処置群の標本RMSEの推定"""
        check_is_fitted(self, "is_fitted_")
        X = check_array(X, dtype=np.float64)
        y = check_array(y, dtype=np.float64, ensure_2d=False)

        # 層化
        labels = self.model.predict(X[:, self.get_selected_features_index()])

        # estimate_y_rmse を呼び出す
        return estimate_y_rmse(
            sample_size_each_cluster=self.sample_size_each_cluster,
            y=y,
            labels=labels,
            true_mean=true_mean,  # true_meanを渡す
            n_trials=self.n_trials,
            random_state=self.random_state,
        )

    def compute_score(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> tuple[float, CLUSTERRING_CLASS]:
        raise NotImplementedError

    def clustering(self, X: NDArray[np.float64]) -> tuple[CLUSTERRING_CLASS, NDArray[np.int_]]:
        # クラスタリング手法がGMMの場合
        if self.clustering_method == "GMM":
            model = GaussianMixture(
                n_components=self.n_clusters,
                random_state=self.random_state,
                init_params="kmeans",
            )
        # クラスタリング手法がKMEANSの場合
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
        return self.selected_features_index  # 選択された特徴量のインデックス

    def get_score_history(self) -> list[float]:
        return self.score_history  # 特徴量数ごとのスコア


class SequentialForwardSelectionVarEstimator(BaseSequentialForwardSelectionEstimator):
    def compute_score(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> tuple[float, BaseEstimator]:
        """特徴量選択の評価関数(大きいほど良い)"""
        model, labels = self.clustering(X)
        sample_size_each_cluster = self.allocator.solve(
            y, labels, self.n_clusters, self.sample_size
        )
        var = compute_y_var(sample_size_each_cluster, y, labels)
        # 最大化のため、符号を反転
        score = -var

        return score, model

    def __str__(self) -> str:
        return f"SFS-{self.clustering_method}-Var"


class SequentialForwardSelectionFEstimator(BaseSequentialForwardSelectionEstimator):
    def compute_score(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> tuple[float, BaseEstimator]:
        """特徴量選択の評価関数(大きいほど良い)"""
        model, _ = self.clustering(X)
        score = model.score(X)

        return score, model

    def __str__(self) -> str:
        return f"SFS-{self.clustering_method}-F"


class SequentialForwardSelectionTEstimator(BaseSequentialForwardSelectionEstimator):
    def compute_score(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> tuple[float, BaseEstimator]:
        """特徴量選択の評価関数(大きいほど良い)"""
        model, labels = self.clustering(X)
        if self.clustering_method != "KMeans":
            warnings.warn(
                "Total Sum of Squares is only available for KMeans clustering. if you use other clustering methods, Unexpected result may be returned."
            )
        # Total Sum of Squares in each cluster
        tss = 0
        for k in np.unique(labels):
            y_k = y[labels == k]
            mean_y_k = np.mean(y_k)
            tss += np.sum((y_k - mean_y_k) ** 2)
        score = -tss

        return score, model

    def __str__(self) -> str:
        return f"SFS-{self.clustering_method}-T"
