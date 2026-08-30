from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class CUPEDEstimator(RegressorMixin, BaseEstimator):  # type: ignore
    def __init__(self, n_trials: int, sample_size: int, random_state: int = 42):
        self.n_trials = n_trials
        self.sample_size = sample_size
        self.random_state = random_state

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "CUPEDEstimator":
        # Validation
        X, y = check_X_y(X, y, dtype=np.float64)

        # 目的変数と最も相関の高い特徴量を選択
        correlations = np.corrcoef(X, y, rowvar=False)[-1, :-1]  # 各特徴量とyの相関係数
        self.most_correlated_var_index = np.argmax(np.abs(correlations))
        self.max_correlation = correlations[self.most_correlated_var_index]

        most_correlated_var = X[:, self.most_correlated_var_index]
        self.x_mean_pop_ = np.mean(most_correlated_var)

        # alpha (theta) の推定
        self.alpha = np.cov(y, most_correlated_var)[0, 1] / np.var(most_correlated_var)
        self.is_fitted_ = True

        return self

    def predict(self, X: NDArray[np.float64], y: NDArray[np.float64], true_mean: float) -> float:
        """標本RMSEの推定（RMSEのみ）"""
        check_is_fitted(self, "is_fitted_")
        X = check_array(X, dtype=np.float64)
        y = check_array(y, dtype=np.float64, ensure_2d=False)

        # 各反復 t における（変換後 y_cuped の）標本平均 μ̂^(t) を並列に計算
        sample_means = Parallel(n_jobs=-1)(
            delayed(self.estimate_y_mean)(X, y, self.random_state + i)
            for i in range(self.n_trials)
        )
        sample_means = np.asarray(sample_means, dtype=float)

        # RMSE = sqrt( (1/T) * sum_t (μ̂^(t) - μ)^2 )
        rmse = float(np.sqrt(np.mean((sample_means - true_mean) ** 2)))

        return rmse

    def estimate_y_mean(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        random_state: Optional[int] = None,
    ) -> float:
        check_is_fitted(self, "is_fitted_")  # モデルが適合されているか確認
        X = check_array(X, dtype=np.float64)  # 入力データを検証
        y = check_array(y, dtype=np.float64, ensure_2d=False)  # y の形状を確認
        most_correlated_var = X[:, self.most_correlated_var_index]
        y_cuped = y - self.alpha * (most_correlated_var - self.x_mean_pop_)

        # サンプリング:
        rng = np.random.RandomState(random_state)
        sample_indices = rng.choice(len(y_cuped), self.sample_size, replace=False)
        y_mean: float = np.mean(y_cuped[sample_indices])

        return y_mean

    def get_selected_features_index(self) -> NDArray[np.int_]:
        return np.array([self.most_correlated_var_index])

    def get_max_correlation(self) -> float:
        return self.max_correlation

    def __str__(self) -> str:
        return "CUPED"
