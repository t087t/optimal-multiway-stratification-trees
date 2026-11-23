"""Simple-random sampling baseline for population-mean estimation."""

from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array


class RandomEstimator(RegressorMixin, BaseEstimator):  # type: ignore
    """Evaluate simple-random population-mean sampling without a fitted model."""

    def __init__(self, n_trials: int = 100, sample_size: int = 50, random_state: int = 42):
        """Store estimator settings without fitting a model.

        Args:
            n_trials: Number of Monte Carlo sampling trials.
            sample_size: Total number of observations to sample without replacement.
            random_state: Initial random seed for model fitting and Monte Carlo sampling.

        Returns:
            None: Settings are stored for subsequent fitting and evaluation.
        """
        self.n_trials = n_trials
        self.sample_size = sample_size
        self.random_state = random_state

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "RandomEstimator":
        """Return the estimator without fitting a model.

        Args:
            X: Unused features, accepted for the common estimator interface.
            y: Unused outcomes, accepted for the common estimator interface.

        Returns:
            RandomEstimator: This unchanged estimator.
        """
        return self

    def predict(self, X: NDArray[np.float64], y: NDArray[np.float64], true_mean: float) -> float:
        """Estimate population-mean RMSE over repeated sampling trials.

        Args:
            X: Unused feature matrix, accepted for the common estimator interface.
            y: Population outcomes of shape (n_samples,), aligned with the feature rows.
            true_mean: Mean of the full population against which estimates are evaluated.

        Returns:
            float: Monte Carlo RMSE, not a vector of row-level predictions.

        Raises:
            ValueError: If inputs have invalid shapes or values, or an allocated draw exceeds its
                stratum population.

        Notes:
            No fitting is needed. Trial i uses random_state + i.
        """
        y = check_array(y, dtype=np.float64, ensure_2d=False)

        y_hats = Parallel(n_jobs=-1)(
            delayed(self.estimate_y_mean)(X, y, self.random_state + i) for i in range(self.n_trials)
        )
        y_hats = np.asarray(y_hats, dtype=float)

        rmse = float(np.sqrt(np.mean((y_hats - true_mean) ** 2)))
        return rmse

    def estimate_y_mean(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        random_state: Optional[int] = None,
    ) -> float:
        """Draw one simple random sample and return its mean.

        Args:
            X: Unused features, accepted for the common estimator interface.
            y: Population outcomes of shape (n_samples,), aligned with the feature rows.
            random_state: Random seed. None creates an unseeded RandomState.

        Returns:
            float: Mean of the sampled outcomes.

        Raises:
            ValueError: If inputs have invalid shapes or values, or an allocated draw exceeds its
                stratum population.
        """
        y = check_array(y, dtype=np.float64, ensure_2d=False)
        rng = np.random.RandomState(random_state)
        sample = rng.choice(y, self.sample_size, replace=False)
        y_mean: float = np.mean(sample)  # type: ignore

        return y_mean

    def __str__(self) -> str:
        """Return the method label used in experiment tables.

        Returns:
            str: Human-readable method name.
        """
        return "Random"
