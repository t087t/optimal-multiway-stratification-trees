"""Export the original depth figures without opening windows."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rich.console import Console
from rich.table import Table


def save(rrr_df, reduction_df, time_df, MAX_DEPTH_LIST, TREE_OUTPUT_DIR):
    console = Console()
    console.print("\n--- [ステップ3] 論文用図表の作成を開始します ---", style="bold cyan")

    # 論文用の日本語マッピング
    allocator_name_map = {
        "Proportional": "比例配分",
        "Optimal": "最適配分",
    }

    # =============================================================================
    # --- 卒論・論文向けのスタイル設定 ---
    # =============================================================================
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

    FIG_DIR = TREE_OUTPUT_DIR / "figures_for_thesis"
    FIG_DIR.mkdir(exist_ok=True, parents=True)

    # 手法ごとのマーカー設定（モノクロ印刷対応）
    marker_map = {
        "OMT(quantile)": "o",  # 円
        "OMT(optbinning)": "s",  # 四角
        "CART": "^",  # 三角
    }

    # -------------------------------------------------------------------------
    # 3a. 木の深さに対する RMSE削減率（RRR）: テストデータのみ
    # -------------------------------------------------------------------------
    for alloc_raw, alloc_jp in allocator_name_map.items():
        # 該当する配分手法のテストデータのみ抽出
        current_df = rrr_df[rrr_df["allocator_name"] == alloc_raw].copy()

        # 図の作成
        fig, ax = plt.subplots(figsize=(8, 6))

        # 折れ線グラフの描画
        # hueとstyleを同時に指定し、markersに辞書を渡すことで手法ごとに形状を固定
        sns.lineplot(
            data=current_df,
            x="max_depth",
            y="rrr_test",
            hue="estimator_name",
            style="estimator_name",
            markers=marker_map,
            markersize=10,
            ax=ax,
        )

        # 軸・ラベル設定（タイトルは無し）
        ax.set_xlabel("木の深さの上限値")
        ax.set_ylabel("RMSE削減率（%）")
        ax.set_xticks(MAX_DEPTH_LIST)

        # グリッド設定
        ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.7)

        # 凡例を上部に横並びで配置（論文の構成に合わせて調整可能）
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles,
            labels=labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),  # 図のすぐ上
            ncol=3,  # 3手法を横に並べる
            frameon=True,
            edgecolor="black",
            fancybox=False,
            columnspacing=1.5,
        )

        outpath = FIG_DIR / f"rrr_vs_depth_test_{alloc_raw}.png"
        plt.savefig(outpath, bbox_inches="tight")
        # Figures are saved by the script; no interactive window is opened.
        plt.close(fig)

    # -------------------------------------------------------------------------
    # 3b. 問題縮小ロジックの効果の表（Rich Table） - コンソール確認用
    # -------------------------------------------------------------------------
    for allocator_name in reduction_df["allocator_name"].unique():
        df_sub_alloc = reduction_df[reduction_df["allocator_name"] == allocator_name].copy()

        for estimator_name in df_sub_alloc["estimator_name"].unique():
            df_sub = df_sub_alloc[df_sub_alloc["estimator_name"] == estimator_name].copy()
            df_sub = df_sub.sort_values("max_features")

            table = Table(
                title=f"問題縮小ロジックの効果サマリー（{allocator_name} / {estimator_name}）",
                show_lines=True,
            )
            table.add_column("深さ", justify="right")
            table.add_column("ルール数（前→後, 削減率）", justify="right")
            table.add_column("制約数（前→後, 削減率）", justify="right")
            table.add_column("計算時間（縮小後）[s]", justify="right")
            table.add_column("計算時間（縮小前）[s]", justify="right")
            table.add_column("高速化（縮小前/後）", justify="right")

            for _, row in df_sub.iterrows():
                rb, ra = row["rules_before"], row["rules_after"]
                cb, ca = row["constraints_before"], row["constraints_after"]

                rules_red = (1 - ra / rb) * 100 if (rb and rb > 0 and not np.isnan(ra)) else np.nan
                cons_red = (1 - ca / cb) * 100 if (cb and cb > 0 and not np.isnan(ca)) else np.nan

                table.add_row(
                    str(int(row["max_features"])),
                    f"{rb:.0f}→{ra:.0f}（{rules_red:.1f}%）"
                    if not np.isnan(rules_red)
                    else f"{rb:.0f}→{ra:.0f}（-）",
                    f"{cb:.0f}→{ca:.0f}（{cons_red:.1f}%）"
                    if not np.isnan(cons_red)
                    else f"{cb:.0f}→{ca:.0f}（-）",
                    f"{row['mip_time_optimized_sec']:.4f}",
                    f"{row['mip_time_unoptimized_sec']:.4f}",
                    f"{row['mip_speedup']:.2f}x" if not np.isnan(row["mip_speedup"]) else "-",
                )
            console.print(table)

    console.print(f"\n✅ 図は {FIG_DIR} にPNG（300dpi）でも保存しました。", style="bold green")
