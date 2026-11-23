"""Internal OMST visualization implementation.

Method bodies are preserved from the original research implementation.
"""

from typing import Any

import graphviz
import numpy as np
from sklearn.utils.validation import check_is_fitted


class RuleVisualizationMixin:
    def _format_condition(self, feature: str, condition: Any) -> str:
        """表示用に条件を分かりやすくフォーマットする"""
        if isinstance(condition, tuple) and len(condition) == 2:
            op, val = condition

            if op == "in":
                cats_str = ", ".join(map(str, val))
                return f"{feature}: {cats_str}"

            if not isinstance(op, str):
                lower, upper = condition
                if lower == -np.inf:
                    return f"{feature} <= {upper:.3f}"
                if upper == np.inf:
                    return f"{feature} > {lower:.3f}"
                return f"{lower:.3f} < {feature} <= {upper:.3f}"

        return f"{feature}: {str(condition)}"

    def _print_selected_rules(self) -> None:
        """選択されたルールを人間が読みやすい形式で表示する"""
        if not self.verbose:
            return

        print("\n" + "=" * 20 + " 最適化により選択されたパス " + "=" * 20)
        rules = self.selected_rules_
        for i, rule in enumerate(rules):
            num_samples = rule.get("train_samples", "N/A")
            std_j = rule.get("std_j", "N/A")
            std_str = f"{std_j:.3f}" if isinstance(std_j, (int, float)) else str(std_j)

            print(f"\n[層 {i + 1}]")
            print(rf"  統計量: 事例数 = {num_samples}, 標準偏差 $\sigma$ = {std_str}")
            print(f"  条件:")
            for feature, condition in rule["features"].items():
                formatted_cond = self._format_condition(feature, condition)
                print(f"    - {formatted_cond}")

    def _visualize_rules(self) -> "graphviz.Digraph":
        """選択されたOMTルールをGraphvizの有向グラフに変換する。"""
        check_is_fitted(self, "is_fitted_")

        dot = graphviz.Digraph(comment="OMT Rules")
        font = "Yu Gothic UI"
        dot.attr(dpi="300")
        dot.attr(
            "node",
            fontname=font,
            fontsize="11",
            margin="0.12",
            shape="box",
            style="rounded,filled",
        )
        dot.attr("edge", fontname=font, fontsize="9", color="#555555")

        root_id = "root"
        dot.node(root_id, "Root", fillcolor="#FFFFFF", fontcolor="black", shape="ellipse")
        created_nodes = {root_id}
        created_edges = set()
        order_map = {feature: i for i, feature in enumerate(self.feature_order_)}

        for i, rule in enumerate(self.selected_rules_):
            current_node_id = root_id
            path_parts = []
            features_sorted = sorted(
                rule["features"].items(),
                key=lambda kv: order_map.get(kv[0], len(order_map)),
            )

            for feature, condition in features_sorted:
                condition_label = self._format_condition(feature, condition)
                path_parts.append(condition_label)
                node_id = f"node_{abs(hash(tuple(path_parts)))}"

                if node_id not in created_nodes:
                    color_index = (abs(hash(feature)) % 9) + 1
                    dot.node(
                        node_id,
                        label=condition_label,
                        colorscheme="pastel19",
                        fillcolor=str(color_index),
                    )
                    created_nodes.add(node_id)

                edge = (current_node_id, node_id)
                if edge not in created_edges:
                    dot.edge(*edge)
                    created_edges.add(edge)

                current_node_id = node_id

            leaf_id = f"leaf_{i}"
            leaf_label = (
                f"Rule {i + 1}\n"
                f"Samples: {rule.get('train_samples', 0)}\n"
                f"Std: {rule.get('std_j', 0.0):.3f}"
            )
            dot.node(leaf_id, label=leaf_label, fillcolor="#FBDD64", color="#000000")
            dot.edge(current_node_id, leaf_id)

        return dot

    def save_tree_png(self, filename: str = "omt_tree.png"):
        """学習済みのOMTルールをPNGファイルとして保存する。"""
        check_is_fitted(self, "is_fitted_")

        if graphviz is None:
            raise ImportError("graphvizが必要")

        base_filename = filename[:-4] if filename.lower().endswith(".png") else filename
        self._visualize_rules().render(base_filename, format="png", cleanup=True, view=False)

        if self.verbose:
            print(f"決定木を '{filename}' に保存")
