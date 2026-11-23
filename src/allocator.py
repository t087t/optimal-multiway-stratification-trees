"""Sample allocation and population-mean estimation for stratified sampling."""

from typing import Optional, Protocol

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray


class AllocatorProtocol(Protocol):
    """Describe an allocator that produces integer counts for consecutive strata."""

    def solve(
        self,
        y: NDArray[np.float64],
        labels: NDArray[np.int_],
        n_clusters: int,
        sample_size: int,
    ) -> NDArray[np.int_]:
        """Describe the integer-allocation interface.

        Args:
            y: Population outcomes of shape (n_samples,), aligned with the feature rows.
            labels: Nonnegative, consecutive stratum labels of shape (n_samples,).
            n_clusters: Number of strata or clusters.
            sample_size: Total number of observations to sample without replacement.

        Returns:
            ndarray: Sample counts of shape (n_clusters,).
        """
        ...


def compute_y_var(
    sample_size_each_cluster: NDArray[np.int_],
    y: NDArray[np.float64],
    labels: NDArray[np.int_],
) -> float:
    """Compute finite-population variance for a stratified mean.

    Args:
        sample_size_each_cluster: Integer allocation per stratum, with shape (n_strata,).
        y: Population outcomes of shape (n_samples,), aligned with the feature rows.
        labels: Nonnegative, consecutive stratum labels of shape (n_samples,).

    Returns:
        float: Analytic variance under the supplied allocation.

    Raises:
        ValueError: If labels contain negative values or array dimensions are incompatible.

    Notes:
        Allocations above a stratum population are capped locally. Strata with zero allocation
        are excluded.
    """
    data_size = len(y)
    size_each_cluster = np.bincount(labels, minlength=len(sample_size_each_cluster))

    clusters = np.arange(len(sample_size_each_cluster))

    if np.any(sample_size_each_cluster > size_each_cluster):
        sample_size_each_cluster = np.minimum(sample_size_each_cluster, size_each_cluster)

    mask = sample_size_each_cluster > 0
    clusters = clusters[mask]
    size_each_cluster = size_each_cluster[mask]
    sample_size_each_cluster_masked = sample_size_each_cluster[mask]

    var_each_cluster = np.array(
        [np.var(y[labels == h], ddof=1) if np.sum(labels == h) > 1 else 0.0 for h in clusters]
    )

    var: float = (1 / data_size**2) * (
        ((size_each_cluster**2 * var_each_cluster) / sample_size_each_cluster_masked).sum()
        - (size_each_cluster * var_each_cluster).sum()
    )

    return var


def estimate_y_mean(
    sample_size_each_cluster: NDArray[np.int_],
    y: NDArray[np.float64],
    labels: NDArray[np.int_],
    random_state: Optional[int] = None,
) -> float:
    """Draw one stratified sample and estimate the population mean.

    Args:
        sample_size_each_cluster: Integer allocation per stratum, with shape (n_strata,).
        y: Population outcomes of shape (n_samples,), aligned with the feature rows.
        labels: Nonnegative, consecutive stratum labels of shape (n_samples,).
        random_state: Random seed. None creates an unseeded RandomState.

    Returns:
        float: Weighted mean over strata with positive allocation.

    Raises:
        ValueError: If inputs have invalid shapes or values, or an allocated draw exceeds its
            stratum population.

    Notes:
        Weights are renormalized after zero-allocation strata are excluded.
    """
    n_clusters = len(sample_size_each_cluster)
    size_each_cluster = np.bincount(labels, minlength=n_clusters)

    if np.any(sample_size_each_cluster > size_each_cluster):
        raise ValueError(
            "Allocated sample size exceeds cluster size.\n"
            f"sample_size_each_cluster = {sample_size_each_cluster}\n"
            f"size_each_cluster      = {size_each_cluster}"
        )

    clusters = np.arange(n_clusters)
    weight_each_cluster = size_each_cluster / size_each_cluster.sum()

    mask = sample_size_each_cluster > 0
    clusters = clusters[mask]
    sample_size_each_cluster = sample_size_each_cluster[mask]
    weight_each_cluster = weight_each_cluster[mask]

    w_sum = weight_each_cluster.sum()
    if w_sum > 0:
        weight_each_cluster = weight_each_cluster / w_sum

    rng = np.random.RandomState(random_state)
    y_mean = 0.0

    for cluster, n_h, weight_h in zip(clusters, sample_size_each_cluster, weight_each_cluster):
        y_h = y[labels == cluster]
        y_h_sample = rng.choice(y_h, n_h, replace=False)
        y_mean += y_h_sample.mean() * weight_h

    return y_mean


def estimate_y_var(
    sample_size_each_cluster: NDArray[np.int_],
    y: NDArray[np.float64],
    labels: NDArray[np.int_],
    n_trials: int = 100,
    random_state: Optional[int] = None,
) -> float:
    """Estimate sampling variance by repeated stratified draws.

    Args:
        sample_size_each_cluster: Integer allocation per stratum, with shape (n_strata,).
        y: Population outcomes of shape (n_samples,), aligned with the feature rows.
        labels: Nonnegative, consecutive stratum labels of shape (n_samples,).
        n_trials: Number of Monte Carlo sampling trials.
        random_state: Initial trial seed. None draws a seed from the global NumPy generator.

    Returns:
        float: Sample variance of trial means, or zero for at most one trial.

    Raises:
        ValueError: If inputs have invalid shapes or values, or an allocated draw exceeds its
            stratum population.

    Notes:
        Trial i uses random_state + i after resolving an unspecified seed.
    """
    if random_state is None:
        random_state = np.random.randint(0, 2**32 - 1)

    y_means = Parallel(n_jobs=1)(
        delayed(estimate_y_mean)(sample_size_each_cluster, y, labels, i + random_state)
        for i in range(n_trials)
    )

    if len(y_means) <= 1:
        var: float = 0.0
    else:
        var: float = np.var(y_means, ddof=1)  # type: ignore

    return var


def estimate_y_rmse(
    sample_size_each_cluster: NDArray[np.int_],
    y: NDArray[np.float64],
    labels: NDArray[np.int_],
    true_mean: float,
    n_trials: int,
    random_state: Optional[int] = None,
) -> float:
    """Estimate population-mean RMSE by repeated stratified draws.

    Args:
        sample_size_each_cluster: Integer allocation per stratum, with shape (n_strata,).
        y: Population outcomes of shape (n_samples,), aligned with the feature rows.
        labels: Nonnegative, consecutive stratum labels of shape (n_samples,).
        true_mean: Mean of the full population against which estimates are evaluated.
        n_trials: Number of Monte Carlo sampling trials.
        random_state: Initial trial seed. None draws a seed from the global NumPy generator.

    Returns:
        float: Square root of the mean squared error across trials.

    Raises:
        ValueError: If inputs have invalid shapes or values, or an allocated draw exceeds its
            stratum population.

    Notes:
        Trial i uses random_state + i; true_mean refers to the full evaluation population.
    """
    if random_state is None:
        random_state = np.random.randint(0, 2**32 - 1)

    y_means_list = Parallel(n_jobs=1)(
        delayed(estimate_y_mean)(sample_size_each_cluster, y, labels, i + random_state)
        for i in range(n_trials)
    )
    y_means_list = np.asarray(y_means_list, dtype=float)

    rmse = float(np.sqrt(np.mean((y_means_list - true_mean) ** 2)))
    return rmse


class ProportionalAllocator:
    """Allocate samples in proportion to population counts using largest remainders."""

    def solve(
        self,
        y: NDArray[np.float64],
        labels: NDArray[np.int_],
        n_clusters: int,
        sample_size: int,
    ) -> NDArray[np.int_]:
        """Allocate integer sample counts using the largest-remainder method.

        Args:
            y: Population outcomes; accepted for the allocator interface but not used.
            labels: Nonnegative, consecutive stratum labels of shape (n_samples,).
            n_clusters: Number of strata or clusters.
            sample_size: Total number of observations to sample without replacement.

        Returns:
            ndarray: Per-stratum allocation of shape (n_clusters,).

        Raises:
            ValueError: If labels contain negative values or input dimensions are invalid.
        """
        cluster_sizes = np.bincount(labels, minlength=n_clusters)
        total_population = cluster_sizes.sum()

        quotas = (cluster_sizes / total_population) * sample_size

        n = np.floor(quotas).astype(int)

        remainders = quotas - n

        n_missing = sample_size - n.sum()

        # Preserve the existing descending argsort order when remainders tie.
        indices_descending_remainder = np.argsort(remainders)[::-1]

        for i in range(n_missing):
            idx = indices_descending_remainder[i]
            n[idx] += 1

        return n

    def __str__(self) -> str:
        """Return the method label used in experiment tables.

        Returns:
            str: Human-readable method name.
        """
        return "Proportional"


class OptimalAllocator:
    """Allocate samples greedily under stratum-specific lower and upper bounds."""

    def __init__(
        self,
        m_value: int,
        M: Optional[NDArray[np.float64]] = None,
    ):
        """Store lower and optional upper allocation bounds.

        Args:
            m_value: Minimum sample count assigned to every stratum.
            M: Per-stratum upper bounds. None uses the stratum population sizes.

        Returns:
            None: Bounds are stored without solving an allocation.
        """
        self.m_value = m_value
        self.M = M

    def solve(
        self,
        y: NDArray[np.float64],
        labels: NDArray[np.int_],
        n_clusters: int,
        sample_size: int,
    ) -> NDArray[np.int_]:
        """Allocate samples by the greatest marginal variance reduction.

        Args:
            y: Population outcomes of shape (n_samples,), aligned with the feature rows.
            labels: Nonnegative, consecutive stratum labels of shape (n_samples,).
            n_clusters: Number of strata or clusters.
            sample_size: Total number of observations to sample without replacement.

        Returns:
            ndarray: Integer allocation, possibly below sample_size if upper bounds prevent further
                draws.

        Raises:
            AssertionError: If the final allocation violates a checked total, lower, or supplied
                upper bound.
            ValueError: If labels or allocation-array dimensions are invalid.

        Notes:
            Stratum labels must be consecutive from zero. The method starts every stratum at m_value
            and increments allocations greedily.
        """
        cluster_sizes = np.bincount(labels, minlength=n_clusters)
        H = len(cluster_sizes)

        S = np.array(
            [np.var(y[labels == h], ddof=1) if np.sum(labels == h) > 1 else 0.0 for h in range(H)]
        )
        d = (cluster_sizes**2) * S
        n = np.full(H, self.m_value)

        M = self.M.copy() if self.M is not None else cluster_sizes.copy()
        I = np.arange(H)  # noqa
        while (n.sum() != sample_size) and len(I) != 0:
            delta = np.zeros(H)
            # Choose the stratum giving the largest marginal variance reduction.
            delta[I] = (d / (n + 1) - d / n)[I]
            h_star = np.argmin(delta[I])
            h_star = I[h_star]

            if n[h_star] + 1 <= M[h_star]:
                n[h_star] = n[h_star] + 1
            else:
                I_ = I.tolist()
                I_ = [i for i in I_ if i != h_star]
                I = np.array(I_)  # type: ignore # noqa

        assert n.sum() <= sample_size, f"Total sample size is over than {sample_size}"
        assert np.all(n >= self.m_value), "Minimum sample size constraint is not satisfied"
        if self.M is not None:
            assert np.all(n <= self.M), "Maximum sample size constraint is not satisfied"

        return n

    def __str__(self) -> str:
        """Return the method label used in experiment tables.

        Returns:
            str: Human-readable method name.
        """
        return "Optimal"


class PostAllocator:
    """Provide post-stratified estimation; direct allocation remains unimplemented."""

    def solve(
        self, y: NDArray[np.float64], labels: NDArray[np.int_], sample_size: int
    ) -> NDArray[np.int_]:
        """Expose the unimplemented post-stratification allocation operation.

        Args:
            y: Population outcomes of shape (n_samples,), aligned with the feature rows.
            labels: Nonnegative, consecutive stratum labels of shape (n_samples,).
            sample_size: Total number of observations to sample without replacement.

        Returns:
            NoReturn: This operation always raises NotImplementedError.

        Raises:
            NotImplementedError: Always; direct allocation is not implemented.
        """
        raise NotImplementedError("PostAllocatorのsolveメソッドは実装されていません")

    @staticmethod
    def estimate_y_mean(
        sample_size: int,
        weight_each_cluster: NDArray[np.float64],
        y: NDArray[np.float64],
        labels: NDArray[np.int_],
        rng: Optional[np.random.RandomState] = None,
    ) -> float:
        """Estimate a post-stratified mean from one simple random sample.

        Args:
            sample_size: Total number of observations to sample without replacement.
            weight_each_cluster: Population weight per stratum, with shape (n_strata,).
            y: Population outcomes of shape (n_samples,), aligned with the feature rows.
            labels: Nonnegative, consecutive stratum labels of shape (n_samples,).
            rng: RandomState shared across draws. None creates a new unseeded generator.

        Returns:
            float: Sum of sampled stratum means times the supplied population weights.

        Raises:
            ValueError: If the requested sample cannot be drawn without replacement.

        Notes:
            Unsampled strata contribute zero; the remaining weights are not renormalized.
        """
        data_size = len(y)
        if rng is None:
            rng = np.random.RandomState()
        sample_indices = rng.choice(data_size, sample_size, replace=False)
        y_sample = y[sample_indices]
        labels_sample = labels[sample_indices]
        unique_labels_sample = set(labels_sample)

        y_mean = 0.0
        for h, weight in enumerate(weight_each_cluster):
            if h not in unique_labels_sample:
                continue
            y_sample_h = y_sample[labels_sample == h]
            y_sample_mean = y_sample_h.mean()
            y_mean += y_sample_mean * weight

        return y_mean

    @classmethod
    def estimate_y_var(
        cls,
        sample_size: int,
        weight_each_cluster: NDArray[np.float64],
        y: NDArray[np.float64],
        labels: NDArray[np.int_],
        n_trials: int = 100,
        random_state: Optional[int] = None,
    ) -> float:
        """Estimate variance from post-stratified simple-random draws.

        Args:
            sample_size: Total number of observations to sample without replacement.
            weight_each_cluster: Population weight per stratum, with shape (n_strata,).
            y: Population outcomes of shape (n_samples,), aligned with the feature rows.
            labels: Nonnegative, consecutive stratum labels of shape (n_samples,).
            n_trials: Number of Monte Carlo sampling trials.
            random_state: Random seed. None creates an unseeded RandomState.

        Returns:
            float: Sample variance of the trial estimates with ddof=1.

        Raises:
            ValueError: If the requested sample cannot be drawn without replacement.

        Notes:
            A shared RandomState advances between trials, rather than being reseeded for each draw.
        """
        y_means = np.zeros(n_trials)
        rng = np.random.RandomState(random_state)
        for i in range(n_trials):
            y_mean = cls.estimate_y_mean(sample_size, weight_each_cluster, y, labels, rng=rng)
            y_means[i] = y_mean

        var: float = np.var(y_means, ddof=1)  # type: ignore

        return var

    def __str__(self) -> str:
        """Return the method label used in experiment tables.

        Returns:
            str: Human-readable method name.
        """
        return "Post"
