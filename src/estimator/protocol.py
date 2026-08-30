from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class EstimatorProtocol(Protocol):
    """
    estimatorのためのProtocol型定義

    fitとpredictメソッドを持つestimatorの型ヒントを定義します。
    """

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "EstimatorProtocol":
        """
        モデルを学習するメソッド

        Parameters
        ----------
        X : NDArray[np.float64]
            特徴量の配列
        y : NDArray[np.float64]
            目標変数の配列

        Returns
        -------
        EstimatorProtocol
            学習済みのestimatorインスタンス自身
        """
        ...

    def predict(self, X: NDArray[np.float64], y: NDArray[np.float64], true_mean: float) -> float:
        """
        予測を行うメソッド

        Parameters
        ----------
        X : NDArray[np.float64]
            特徴量の配列
        y : NDArray[np.float64]
            目標変数の配列
        true_mean : float
            母集団の真の平均値

        Returns
        -------
        float : RMSEの推定値
        """
        ...
