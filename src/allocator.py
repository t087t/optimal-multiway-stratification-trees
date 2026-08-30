from typing import Optional, Protocol

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray


class AllocatorProtocol(Protocol):
    """
    allocatorのためのProtocol型定義

    solveメソッドを持つallocatorの型ヒントを定義します。
    """

    def solve(
        self, y: NDArray[np.float64], labels: NDArray[np.int_], n_clusters: int, sample_size: int
    ) -> NDArray[np.int_]:
        """標本配分を解く

        Args:
            y (NDArray[np.float64]): 目的変数 (N)
            labels (NDArray[np.int_]): クラスタラベル (N)
            n_clusters (int): クラスタ数
            sample_size (int): 総標本数

        Returns:
            NDArray[np.int_]: 各クラスタの標本数 (H, )

        Note:
            H: クラスタ数
        """
        ...


# Allocatorで共通して使用するユーティリティ関数
def compute_y_var(
    sample_size_each_cluster: NDArray[np.int_],
    y: NDArray[np.float64],
    labels: NDArray[np.int_],
) -> float:
    """母集団と標本数が与えられたときの目的変数の分散を計算する

    Args:
        sample_size_each_cluster (NDArray[np.int_]): クラスタ毎の標本数 (H, )
        y (NDArray[np.float64]): 目的変数 (N)
        labels (NDArray[np.int_]): クラスタラベル (N)

    Returns:
        float: 目的変数の分散
    """
    data_size = len(y)
    size_each_cluster = np.bincount(labels, minlength=len(sample_size_each_cluster))
    # クラスタ名は0から連番となっていることを仮定していることに注意
    clusters = np.arange(len(sample_size_each_cluster))

    # クラスタサイズよりも標本数が多い場合は警告を出力して小さい方を採用
    if np.any(sample_size_each_cluster > size_each_cluster):
        sample_size_each_cluster = np.minimum(sample_size_each_cluster, size_each_cluster)

    # サンプルサイズが0のクラスタを除外
    mask = sample_size_each_cluster > 0
    clusters = clusters[mask]
    size_each_cluster = size_each_cluster[mask]
    sample_size_each_cluster_masked = sample_size_each_cluster[mask]
    # クラスタ毎の目的変数の分散（サンプル数が1以下の場合は0とする）
    var_each_cluster = np.array(
        [np.var(y[labels == h], ddof=1) if np.sum(labels == h) > 1 else 0.0 for h in clusters]
    )

    # 目的変数の分散を推定
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
    """標本数が与えられたときの目的変数の平均を推定する

    Args:
        sample_size_each_cluster (NDArray[np.int_]): クラスタ毎の標本数 (H, )
        y (NDArray[np.float64]): 目的変数 (N)
        labels (NDArray[np.int_]): クラスタラベル (N)

    Returns:
        float: 目的変数の平均
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

    # サンプルサイズが0のクラスタを除外
    mask = sample_size_each_cluster > 0
    clusters = clusters[mask]
    sample_size_each_cluster = sample_size_each_cluster[mask]
    weight_each_cluster = weight_each_cluster[mask]

    # 重みを再正規化
    w_sum = weight_each_cluster.sum()
    if w_sum > 0:
        weight_each_cluster = weight_each_cluster / w_sum

    rng = np.random.RandomState(random_state)
    y_mean = 0.0

    for cluster, n_h, weight_h in zip(
        clusters, sample_size_each_cluster, weight_each_cluster
    ):
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
    """標本数が与えられたときの目的変数の分散を推定する

    Args:
        sample_size_each_cluster (NDArray[np.int_]): クラスタ毎の標本数 (H, )
        weight_each_cluster (NDArray[np.float64]): クラスタ毎の重み (H, )
        y (NDArray[np.float64]): 目的変数 (N)
        labels (NDArray[np.int_]): クラスタラベル (N)
        n_trials (int, optional): 試行回数 (default: 100)
        random_state (int, optional): 乱数シード (default: None)

    Returns:
        float: 目的変数の分散
    """
    if random_state is None:
        random_state = np.random.randint(0, 2**32 - 1)

    y_means = Parallel(n_jobs=1)(
        delayed(estimate_y_mean)(sample_size_each_cluster, y, labels, i + random_state)
        for i in range(n_trials)
    )

    # 分散を計算（サンプル数が1以下の場合は0とする）
    if len(y_means) <= 1:
        var: float = 0.0
    else:
        var: float = np.var(y_means, ddof=1)  # type: ignore

    return var


'''def estimate_y_rmse_and_components(
    sample_size_each_cluster: NDArray[np.int_],
    y: NDArray[np.float64],
    labels: NDArray[np.int_],
    true_mean: float,
    n_trials: int,
    random_state: Optional[int] = None,
) -> dict:
    """
    シミュレーションを繰り返し、分散、バイアス、MSE、RMSEを計算する。
    """
    # 標本平均のリストを取得
    y_means_list = Parallel(n_jobs=1)(
        delayed(estimate_y_mean)(sample_size_each_cluster, y, labels, i + random_state)
        for i in range(n_trials)
    )
    y_means_list = np.array(y_means_list)

    # 分散とバイアスを計算
    variance = np.var(y_means_list, ddof=1)  # type: ignore
    bias = np.mean(y_means_list) - true_mean

    # MSEとRMSEを計算
    mse = bias**2 + variance
    rmse = np.sqrt(mse)

    return {"rmse": rmse, "variance": variance, "bias": bias}'''


def estimate_y_rmse(
    sample_size_each_cluster: NDArray[np.int_],
    y: NDArray[np.float64],
    labels: NDArray[np.int_],
    true_mean: float,
    n_trials: int,
    random_state: Optional[int] = None,
) -> float:
    """シミュレーションを繰り返し、RMSEのみを推定する。

    RMSE = sqrt( (1/T) * sum_{t=1..T} (mu_hat^{(t)} - mu)^2 )
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
    """
    各クラスタ数に比例した標本数で分割するアロケーター
    最大剰余法（Hamilton Method）を使用して整数問題を解決します。
    """

    def solve(
        self, y: NDArray[np.float64], labels: NDArray[np.int_], n_clusters: int, sample_size: int
    ) -> NDArray[np.int_]:
        """各クラスタ数に比例した標本数で分割する（最大剰余法）"""

        # 1. 各クラスタの母集団サイズ (N_i) を計算
        cluster_sizes = np.bincount(labels, minlength=n_clusters)
        total_population = cluster_sizes.sum()

        # 2. 理想的な配分数 (quota) を計算
        # q_i = (N_i / N) * n
        quotas = (cluster_sizes / total_population) * sample_size

        # 3. 整数部 (floor) を取得して初期配分とする
        n = np.floor(quotas).astype(int)

        # 4. 剰余 (remainder) を計算
        # r_i = q_i - floor(q_i)
        remainders = quotas - n

        # 5. 合計が sample_size になるように不足分 (k) を計算
        n_missing = sample_size - n.sum()

        # 6. 剰余が大きい順に上位 k 個のクラスタに +1 する
        # argsortは昇順なので、[::-1]で降順（大きい順）にする
        indices_descending_remainder = np.argsort(remainders)[::-1]

        # 上位 k 個のインデックスに対して割り当てを増やす
        # n_missing は必ず 0 以上の整数かつクラスタ数以下になります
        for i in range(n_missing):
            idx = indices_descending_remainder[i]
            n[idx] += 1

        return n

    def __str__(self) -> str:
        return "Proportional"


class OptimalAllocator:
    """
    最適な標本配分を行うアロケーター
    """

    def __init__(
        self,
        m_value: int,  # 標本サイズ下限
        M: Optional[
            NDArray[np.float64]
        ] = None,  # 標本サイズ上限 #Optional(Noneである可能性がある)
    ):
        self.m_value = m_value
        self.M = M

    def solve(
        self,
        y: NDArray[np.float64],
        labels: NDArray[np.int_],
        n_clusters: int,
        sample_size: int,
    ) -> NDArray[np.int_]:
        cluster_sizes = np.bincount(labels, minlength=n_clusters)
        H = len(cluster_sizes)  # クラスタラベルは0から連番となっていることを仮定していることに注意
        # クラスタ毎の目的変数のvariance (H, )（サンプル数が1以下の場合は0とする）
        S = np.array(
            [np.var(y[labels == h], ddof=1) if np.sum(labels == h) > 1 else 0.0 for h in range(H)]
        )
        d = (cluster_sizes**2) * S  # (H, )
        n = np.full(H, self.m_value)  # (H, )

        M = self.M.copy() if self.M is not None else cluster_sizes.copy()
        I = np.arange(H)  # noqa
        while (n.sum() != sample_size) and len(I) != 0:
            delta = np.zeros(H)
            delta[I] = (d / (n + 1) - d / n)[I]
            h_star = np.argmin(delta[I])
            h_star = I[h_star]

            if n[h_star] + 1 <= M[h_star]:
                n[h_star] = n[h_star] + 1
            else:
                # Iの要素h_starを削除
                I_ = I.tolist()
                I_ = [i for i in I_ if i != h_star]
                I = np.array(I_)  # type: ignore # noqa

        # 制約チェック
        assert n.sum() <= sample_size, f"Total sample size is over than {sample_size}"
        assert np.all(n >= self.m_value), "Minimum sample size constraint is not satisfied"
        if self.M is not None:
            assert np.all(n <= self.M), "Maximum sample size constraint is not satisfied"

        return n

    def __str__(self) -> str:
        return "Optimal"


class PostAllocator:
    """
    事後層化を行うアロケーター
    """

    def solve(
        self, y: NDArray[np.float64], labels: NDArray[np.int_], sample_size: int
    ) -> NDArray[np.int_]:
        """事後層化による標本配分を行う"""
        # 実装は省略されていますが、AllocatorProtocolに準拠するためにsolveメソッドを追加
        raise NotImplementedError("PostAllocatorのsolveメソッドは実装されていません")

    @staticmethod
    def estimate_y_mean(
        sample_size: int,
        weight_each_cluster: NDArray[np.float64],
        y: NDArray[np.float64],
        labels: NDArray[np.int_],
        rng: Optional[np.random.RandomState] = None,
    ) -> float:
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
                # サンプルに含まれないクラスタはスキップ
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
        y_means = np.zeros(n_trials)
        rng = np.random.RandomState(random_state)
        for i in range(n_trials):
            y_mean = cls.estimate_y_mean(sample_size, weight_each_cluster, y, labels, rng=rng)
            y_means[i] = y_mean

        var: float = np.var(y_means, ddof=1)  # type: ignore

        return var

    def __str__(self) -> str:
        return "Post"
