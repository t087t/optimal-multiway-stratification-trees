# CART.py

# --- ライブラリのインポート ---
from typing import Optional

import graphviz
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor, export_graphviz, export_text
from sklearn.utils.validation import check_is_fitted, check_X_y

# 外部モジュール
from src.allocator import AllocatorProtocol, estimate_y_rmse, estimate_y_var

# --- CARTEstimator クラス ---


class CARTEstimator(RegressorMixin, BaseEstimator):  # type: ignore
    """
    CART (DecisionTreeRegressor) を用いてデータを層化し、
    指定されたアロケータに基づいて標本RMSEを推定するEstimator。
    """

    def __init__(
        self,
        allocator: AllocatorProtocol,
        sample_size: int,
        max_leaf_nodes: Optional[int] = None,
        max_depth: Optional[int] = None,
        n_trials: int = 100,
        random_state: int = 0,
        min_samples_leaf: int = 1,

    ):
        self.allocator = allocator
        self.sample_size = sample_size
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.n_trials = n_trials
        self.random_state = random_state
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "CARTEstimator":
        """
        データからCARTモデルを学習し、層化とサンプルサイズ配分を決定します。
        """
        X_df_or_array = X  # XがDFかArrayか保持しておく
        X, y = check_X_y(X, y, y_numeric=True)

        # 1. CARTモデルの学習
        self.model_ = DecisionTreeRegressor(
            max_leaf_nodes=self.max_leaf_nodes,
            max_depth=self.max_depth,
            random_state=self.random_state,
            min_samples_leaf=self.min_samples_leaf,
        )
        self.model_.fit(X, y)

        # 2. 学習データでの層化
        labels_from_tree = self.model_.apply(X)

        # 3. 葉ノードIDを0からの連番ラベルに変換
        unique_leaf_ids = np.unique(labels_from_tree)
        self.n_leaves_ = len(unique_leaf_ids)
        self.leaf_id_map_ = {leaf_id: i for i, leaf_id in enumerate(unique_leaf_ids)}
        labels = np.array([self.leaf_id_map_[label] for label in labels_from_tree])

        # 4. 各層へのサンプルサイズ割り当て
        self.sample_size_each_cluster = self.allocator.solve(
            y=y,
            labels=labels,
            n_clusters=self.n_leaves_,
            sample_size=self.sample_size,
        )

        self.is_fitted_ = True

        try:
            # Xがpandas DataFrameの場合、列名を取得
            feature_names = X_df_or_array.columns.tolist()
        except AttributeError:
            # Xがnumpy arrayの場合、'feature_0', 'feature_1', ... のような名前を付ける
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        tree_rules = export_text(self.model_, feature_names=feature_names)
        print("--- CART Stratification Rules ---")
        print(tree_rules)
        print("---------------------------------")

        return self

    def predict(self, X: NDArray[np.float64], y: NDArray[np.float64], true_mean: float) -> float:
        """学習した層化戦略に基づいて、処置群の標本RMSEを推定します。"""
        check_is_fitted(self, "is_fitted_")
        X, y = check_X_y(X, y, y_numeric=True)

        leaf_ids_pred = self.model_.apply(X)
        labels = np.array([self.leaf_id_map_.get(label, -1) for label in leaf_ids_pred])

        # estimate_y_rmse を用いてRMSEを推定
        return estimate_y_rmse(
            sample_size_each_cluster=self.sample_size_each_cluster,
            y=y,
            labels=labels,
            true_mean=true_mean,  # true_meanを渡す
            n_trials=self.n_trials,
            random_state=self.random_state,
        )

    def save_tree_png(self, feature_names: list, filename: str = "cart_tree.png"):
        """
        学習済みの決定木をPNGファイルとして保存します。

        Parameters
        ----------
        feature_names : list
            特徴量の名前のリスト。
        filename : str, optional
            保存するファイル名, by default "cart_tree.png"
        """
        check_is_fitted(self)

        font = "Yu Gothic UI Bold"

        # 決定木をDOT形式のデータに変換
        dot_data = export_graphviz(
            self.model_,
            out_file=None,
            feature_names=feature_names,
            filled=True,
            rounded=True,
            special_characters=True,
            fontname=font
        )

        dot_data = dot_data.replace("samples =", "事例数: ")
        dot_data = dot_data.replace("value =", "予測値: ")
        dot_data = dot_data.replace("squared_error =", "平均二乗誤差: ")

        settings = (
            f'graph [dpi=300, labelfontname="{font}", labelfontsize=10, margin=0.2];\n'
            f'    node [fontname="{font}", fontsize=9];'
        )
        dot_data = dot_data.replace('graph [', settings + '\n    graph [', 1)

        # DOTデータからグラフを生成し、PNGとして保存
        graph = graphviz.Source(dot_data)
        # .png拡張子を除いた部分をファイル名としてrenderに渡す
        graph.render(filename.rsplit('.', 1)[0], format="png", cleanup=True)
        print(f"決定木を '{filename}' に保存しました。")

    def __str__(self) -> str:
        return "CART"
