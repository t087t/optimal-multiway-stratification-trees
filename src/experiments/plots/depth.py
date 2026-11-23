"""Figure export and reduction summaries for the depth experiment."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rich.console import Console
from rich.table import Table


def save(rrr_df, reduction_df, time_df, MAX_DEPTH_LIST, TREE_OUTPUT_DIR):
    """Save depth-versus-RMSE plots and print problem-reduction tables.

    Args:
        rrr_df (pandas.DataFrame): Depth-sweep table containing train/test RMSE reduction rates.
        reduction_df (pandas.DataFrame): Rule/constraint counts and reduced/unreduced solver
            timings.
        time_df (pandas.DataFrame): Depth-sweep timing table; retained in the interface but
            unused here.
        MAX_DEPTH_LIST (list[int]): Depth values used for axis ticks.
        TREE_OUTPUT_DIR (pathlib.Path): Depth-experiment directory containing the figure
            subdirectory.

    Returns:
        None: PNGs are saved in figures_for_thesis and reduction summaries are printed.

    Raises:
        OSError: If the output files cannot be written.
        KeyError: If required reduction or RMSE-result columns are missing.
        ValueError: If plotting values or table fields are invalid.
    """
    console = Console()
    console.print("\n--- [ステップ3] 論文用図表の作成を開始します ---", style="bold cyan")

    allocator_name_map = {
        "Proportional": "比例配分",
        "Optimal": "最適配分",
    }

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

    # Distinct markers keep methods distinguishable in grayscale printouts.
    marker_map = {
        "OMT(quantile)": "o",
        "OMT(optbinning)": "s",
        "CART": "^",
    }

    for alloc_raw, alloc_jp in allocator_name_map.items():
        current_df = rrr_df[rrr_df["allocator_name"] == alloc_raw].copy()

        fig, ax = plt.subplots(figsize=(8, 6))

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

        ax.set_xlabel("木の深さの上限値")
        ax.set_ylabel("RMSE削減率（%）")
        ax.set_xticks(MAX_DEPTH_LIST)

        ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.7)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles,
            labels=labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=3,
            frameon=True,
            edgecolor="black",
            fancybox=False,
            columnspacing=1.5,
        )

        outpath = FIG_DIR / f"rrr_vs_depth_test_{alloc_raw}.png"
        plt.savefig(outpath, bbox_inches="tight")

        plt.close(fig)

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
