from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array


class RandomEstimator(RegressorMixin, BaseEstimator):  # type: ignore
    def __init__(self, n_trials: int = 100, sample_size: int = 50, random_state: int = 42):
        self.n_trials = n_trials
        self.sample_size = sample_size
        self.random_state = random_state

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "RandomEstimator":
        return self

    def predict(self, X: NDArray[np.float64], y: NDArray[np.float64], true_mean: float) -> float:
        """標本RMSEを計算"""
        y = check_array(y, dtype=np.float64, ensure_2d=False)

        # 標本平均を並列して計算
        y_hats = Parallel(n_jobs=-1)(
            delayed(self.estimate_y_mean)(X, y, self.random_state + i)
            for i in range(self.n_trials)
        )
        y_hats = np.asarray(y_hats, dtype=float)

        # RMSE = sqrt( mean( (mu_hat - mu)^2 ) )
        rmse = float(np.sqrt(np.mean((y_hats - true_mean) ** 2)))
        return rmse

    def estimate_y_mean(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        random_state: Optional[int] = None,
    ) -> float:
        """標本平均を計算"""
        y = check_array(y, dtype=np.float64, ensure_2d=False)
        rng = np.random.RandomState(random_state)
        sample = rng.choice(y, self.sample_size, replace=False)
        y_mean: float = np.mean(sample)  # type: ignore

        return y_mean

    def __str__(self) -> str:
        return "Random"
