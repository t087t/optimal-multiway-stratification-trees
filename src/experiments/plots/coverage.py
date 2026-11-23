"""Export the original coverage figures without opening windows."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich import print


def save(results_df, min_coverage_values, allocator_name_map, OUTPUT_DIR):
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "IPAexGothic",
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "axes.labelpad": 10,
            "lines.linewidth": 2.5,
        }
    )

    FIG_DIR = OUTPUT_DIR / "thesis_figures"
    FIG_DIR.mkdir(exist_ok=True, parents=True)

    # 出力対象のメトリクス設定
    metrics = [
        ("mse_test", "MSE", "mse"),
        ("variance_test", "分散", "variance"),
        ("bias_test_sq", "バイアスの二乗", "bias2"),
    ]

    # モノクロ印刷でも判別可能なマーカー設定
    marker_map = {
        "分位点ビニング": "o",  # 円
        "最適ビニング": "s",  # 四角
    }

    # 描画ループ
    for alloc_raw, alloc_jp in allocator_name_map.items():
        current_df = results_df[results_df["標本配分"] == alloc_jp].copy()

        # 被覆率の並びを固定
        current_df["min_coverage"] = pd.Categorical(
            current_df["min_coverage"], categories=min_coverage_values, ordered=True
        )

        for col_name, label_jp, file_suffix in metrics:
            # 図の作成（1枚ずつ）
            fig, ax = plt.subplots(figsize=(6, 5))

            # 折れ線グラフの描画
            sns.lineplot(
                data=current_df,
                x="min_coverage",
                y=col_name,
                hue="ビニング手法",
                style="ビニング手法",
                markers=marker_map,
                markersize=10,
                ax=ax,
            )

            # 軸設定
            ax.set_xlabel("被覆率の下限値")
            ax.set_ylabel(label_jp)
            ax.invert_xaxis()  # 1.0(左) -> 0.8(右) の順にする場合
            ax.grid(True, which="major", linestyle="--", alpha=0.7)

            # 凡例を上部に横並びで配置
            # ncol=2 で2列(横並び)にする。loc='lower center' で指定座標の下端を合わせる
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles=handles,
                labels=labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.01),  # 図の枠外、すぐ上
                ncol=2,
                frameon=True,
                edgecolor="black",
                fancybox=False,
                columnspacing=1.5,
            )

            # タイトルは LaTeX の caption で入れるため削除（空文字）
            ax.set_title("")
            # Figures are saved by the script; no interactive window is opened.

            # 保存
            out_filename = f"omt_test_{file_suffix}_{alloc_raw}.png"
            plt.savefig(FIG_DIR / out_filename, bbox_inches="tight")
            plt.close(fig)

    print(f"\n✅ 論文用図表（計6枚）を {FIG_DIR} に保存しました。")
