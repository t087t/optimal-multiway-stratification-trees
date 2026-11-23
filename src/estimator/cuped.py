"""CUPED covariate adjustment and sampling-based RMSE evaluation."""

from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class CUPEDEstimator(RegressorMixin, BaseEstimator):  # type: ignore
    """Reduce sampling error using one fitted, correlated covariate adjustment.

    Attributes:
        most_correlated_var_index: Input-column index of the selected covariate.
        max_correlation: Signed training correlation of the selected covariate.
        x_mean_pop_: Training-population mean of the selected covariate.
        alpha: Fitted coefficient used to adjust sampled outcomes.
    """

    def __init__(self, n_trials: int, sample_size: int, random_state: int = 42):
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

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "CUPEDEstimator":
        """Select the most correlated covariate and fit the CUPED adjustment.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Population outcomes of shape (n_samples,), aligned with the feature rows.

        Returns:
            CUPEDEstimator: This fitted estimator.

        Raises:
            ValueError: If training inputs or model parameters are invalid.
        """
        X, y = check_X_y(X, y, dtype=np.float64)

        correlations = np.corrcoef(X, y, rowvar=False)[-1, :-1]
        self.most_correlated_var_index = np.argmax(np.abs(correlations))
        self.max_correlation = correlations[self.most_correlated_var_index]

        most_correlated_var = X[:, self.most_correlated_var_index]
        self.x_mean_pop_ = np.mean(most_correlated_var)

        # Retain the covariance/variance normalization used in the original experiments.
        self.alpha = np.cov(y, most_correlated_var)[0, 1] / np.var(most_correlated_var)
        self.is_fitted_ = True

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
            Uses the fitted coefficient and training covariate mean for the CUPED adjustment.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(X, dtype=np.float64)
        y = check_array(y, dtype=np.float64, ensure_2d=False)

        sample_means = Parallel(n_jobs=-1)(
            delayed(self.estimate_y_mean)(X, y, self.random_state + i) for i in range(self.n_trials)
        )
        sample_means = np.asarray(sample_means, dtype=float)

        rmse = float(np.sqrt(np.mean((sample_means - true_mean) ** 2)))

        return rmse

    def estimate_y_mean(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        random_state: Optional[int] = None,
    ) -> float:
        """Draw a simple random sample of CUPED-adjusted outcomes.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Population outcomes of shape (n_samples,), aligned with the feature rows.
            random_state: Random seed. None creates an unseeded RandomState.

        Returns:
            float: Mean of sampled outcomes after the fitted covariate adjustment.

        Raises:
            NotFittedError: If the estimator has not been fitted.
            ValueError: If inputs have invalid shapes or values, or an allocated draw exceeds its
                stratum population.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(X, dtype=np.float64)
        y = check_array(y, dtype=np.float64, ensure_2d=False)
        most_correlated_var = X[:, self.most_correlated_var_index]
        y_cuped = y - self.alpha * (most_correlated_var - self.x_mean_pop_)

        rng = np.random.RandomState(random_state)
        sample_indices = rng.choice(len(y_cuped), self.sample_size, replace=False)
        y_mean: float = np.mean(y_cuped[sample_indices])

        return y_mean

    def get_selected_features_index(self) -> NDArray[np.int_]:
        """Return the index of the selected CUPED covariate.

        Returns:
            ndarray: One-element array containing the covariate index.

        Raises:
            AttributeError: If the covariate has not been selected by fit.
        """
        return np.array([self.most_correlated_var_index])

    def get_max_correlation(self) -> float:
        """Return the signed correlation of the selected covariate.

        Returns:
            float: Training correlation between the selected feature and the outcome.

        Raises:
            AttributeError: If fit has not computed the correlation.
        """
        return self.max_correlation

    def __str__(self) -> str:
        """Return the method label used in experiment tables.

        Returns:
            str: Human-readable method name.
        """
        return "CUPED"
