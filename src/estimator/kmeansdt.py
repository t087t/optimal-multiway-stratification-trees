import re

import graphviz
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from src.allocator import AllocatorProtocol, estimate_y_rmse


class KMeansDTEstimator(RegressorMixin, BaseEstimator):  # type: ignore
    """KMeans + DecisionTree による層化推定器（hir-19-186 仕様）

    - 学習: KMeansでクラスタ（教師ラベル）→ そのクラスタIDを教師に DecisionTreeClassifier を学習
    - 層化(strata): 決定木の「葉(leaf)」を層として用いる
    - 解釈用: 各葉に対して majority cluster（多数派クラスタ）を付与できる
    """

    def __init__(
        self,
        n_clusters: int,
        allocator: AllocatorProtocol,
        sample_size: int,
        n_trials: int = 100,
        random_state: int = 0,
        max_leaf_nodes: int = 8,
    ):
        self.n_clusters = n_clusters
        self.allocator = allocator
        self.n_trials = n_trials
        self.sample_size = sample_size
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes

    # -----------------------------
    # Fit
    # -----------------------------
    def fit(
        self,
        X_std: NDArray[np.float64],
        y: NDArray[np.float64],
        X_orig: NDArray[np.float64],
    ) -> "KMeansDTEstimator":
        """KMeansは標準化データ、決定木は非標準化データで学習する。"""
        X_std, y = check_X_y(X_std, y, dtype=np.float64)
        X_orig = check_array(X_orig, dtype=np.float64)

        # 1) KMeans でクラスタリング（教師ラベル）
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init="auto",
        )
        kmeans_labels: NDArray[np.int_] = self.kmeans_.fit_predict(X_std).astype(int)

        # 2) 決定木で「クラスタIDを予測」するように学習
        self.dt_model_ = DecisionTreeClassifier(
            random_state=self.random_state,
            max_leaf_nodes=self.max_leaf_nodes,
        )
        self.dt_model_.fit(X_orig, kmeans_labels)

        # 3) strata = 葉ID（leaf id）として学習データを割り当て
        leaf_node_ids: NDArray[np.int_] = self.dt_model_.apply(X_orig).astype(int)

        # 3-1) leaf node id は木の内部IDなので、0..H-1にリマップ（連番前提のため）
        unique_leaf_ids = np.unique(leaf_node_ids)
        self.leaf_id_to_stratum_ = {leaf_id: i for i, leaf_id in enumerate(unique_leaf_ids)}
        self.stratum_to_leaf_id_ = {i: leaf_id for leaf_id, i in self.leaf_id_to_stratum_.items()}

        strata_labels = np.vectorize(self.leaf_id_to_stratum_.get)(leaf_node_ids).astype(int)
        self.n_strata_ = int(unique_leaf_ids.size)

        # 4) strata（=葉）単位で標本配分を計算（←ここが「allocatorにleaf_id(=strata)を渡す」部分）
        self.sample_size_each_stratum_ = self.allocator.solve(
            y=y,
            labels=strata_labels,
            n_clusters=self.n_strata_,
            sample_size=self.sample_size,
        )

        # 5) 解釈用：各stratum（葉）に多数派クラスタを付与
        majority = np.zeros(self.n_strata_, dtype=int)
        for h in range(self.n_strata_):
            km_in_h = kmeans_labels[strata_labels == h]
            if km_in_h.size == 0:
                majority[h] = -1
            else:
                counts = np.bincount(km_in_h, minlength=self.n_clusters)
                majority[h] = int(np.argmax(counts))
        self.majority_cluster_each_stratum_ = majority

        self.is_fitted_ = True
        return self

    # -----------------------------
    # Predict helpers
    # -----------------------------
    def predict_strata(self, X_orig: NDArray[np.float64]) -> NDArray[np.int_]:
        """各サンプルが属する strata（=葉）を 0..H-1 の連番で返す。"""
        check_is_fitted(self, attributes=["dt_model_", "leaf_id_to_stratum_", "n_strata_"])
        X_orig = check_array(X_orig, dtype=np.float64)

        leaf_node_ids = self.dt_model_.apply(X_orig).astype(int)

        # 既存の木の葉IDなので基本的に必ず辞書に存在するはず
        try:
            strata_labels = np.vectorize(self.leaf_id_to_stratum_.__getitem__)(leaf_node_ids).astype(int)
        except KeyError as e:
            raise ValueError(
                "未知のleaf_idが出ました。通常は起こりませんが、モデル破損/不整合の可能性があります。"
            ) from e
        return strata_labels

    def predict_cluster(self, X_orig: NDArray[np.float64]) -> NDArray[np.int_]:
        """各サンプルの代表クラスタ（多数派クラスタ）を返す（解釈/可視化用）。"""
        check_is_fitted(self, attributes=["majority_cluster_each_stratum_"])
        strata = self.predict_strata(X_orig)
        return self.majority_cluster_each_stratum_[strata]

    # -----------------------------
    # Main predict (RMSE等の推定)
    # -----------------------------
    def predict(
        self,
        X_orig: NDArray[np.float64],
        y: NDArray[np.float64],
        true_mean: float,
    ) -> float:
        """学習した strata（=葉）に基づいて、標本平均推定の RMSE 等をシミュレーションで返す。"""
        check_is_fitted(self, attributes=["sample_size_each_stratum_"])
        X_orig = check_array(X_orig, dtype=np.float64)
        y = check_array(y, dtype=np.float64, ensure_2d=False)

        labels = self.predict_strata(X_orig)

        return estimate_y_rmse(
            sample_size_each_cluster=self.sample_size_each_stratum_,
            y=y,
            labels=labels,
            true_mean=true_mean,
            n_trials=self.n_trials,
            random_state=self.random_state,
        )

    # -----------------------------
    # Utilities
    # -----------------------------
    @staticmethod
    def lighten_color(hex_color, factor=0.4):
        """
        16進数の色を白と混ぜて薄くする。
        $C_{new} = C_{old} + (255 - C_{old}) \times factor$
        factor: 0.0 (変化なし) ～ 1.0 (真っ白)
        """
        hex_color = hex_color.lstrip('#')
        # RGBに変換
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        # 白とブレンド
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)

        return f"#{r:02x}{g:02x}{b:02x}"

    def save_decision_tree(self, feature_names: list, filename: str = "decision_tree.png") -> None:
        check_is_fitted(self, attributes=["dt_model_"])

        font = "Yu Gothic UI Bold"
        # export_graphviz から fontsize=12 を削除
        dot_data = export_graphviz(
            self.dt_model_,
            out_file=None,
            feature_names=feature_names,
            class_names=[f"クラスタ {i}" for i in range(self.n_clusters)],
            filled=True,
            rounded=True,
            special_characters=True,
            fontname=font
        )

        # テキストの置換
        dot_data = dot_data.replace("samples =", "事例数: ")
        dot_data = dot_data.replace("value =", "クラス別事例数: ")
        dot_data = dot_data.replace("gini =", "ジニ係数: ")
        dot_data = dot_data.replace("class =", "予測クラス: ")

        # グラフ全体とノード個別のフォントサイズ設定を注入
        # node [fontsize=12] を追加することで、箱の中の文字サイズを制御します
        settings = (
            f'graph [dpi=300, labelfontname="{font}", labelfontsize=10, margin=0.2];\n'
            f'    node [fontname="{font}", fontsize=12, margin="0.3,0.15"];'
        )
        dot_data = dot_data.replace('graph [', settings + '\n    graph [', 1)

        # 色を薄くする処理
        lighten_factor = 0.2
        dot_data = re.sub(
            r'fillcolor="(#[A-Fa-f0-9]{6})"',
            lambda m: f'fillcolor="{self.lighten_color(m.group(1), factor=lighten_factor)}"',
            dot_data
        )

        # 保存実行
        graph = graphviz.Source(dot_data)
        graph.render(filename.replace(".png", ""), format="png", cleanup=True)
        print(f"決定木を '{filename}' に保存しました。")

    def __str__(self) -> str:
        return "KMeans-DT"
