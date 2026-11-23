"""Figure export for the coverage-bound experiment."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich import print


def save(results_df, min_coverage_values, allocator_name_map, OUTPUT_DIR):
    """Save test MSE, variance, and squared-bias plots by allocator.

    Args:
        results_df (pandas.DataFrame): Coverage-sweep table containing bias, variance, and MSE
            columns.
        min_coverage_values (list[float]): Coverage bounds in the order used by the experiment.
        allocator_name_map (dict[str, str]): Mapping from allocator identifiers to display
            labels.
        OUTPUT_DIR (pathlib.Path): Directory in which the comparison or coverage figures are
            saved.

    Returns:
        None: Six PNG figures are written to the thesis_figures subdirectory.

    Raises:
        OSError: If the output files cannot be written.
        KeyError: If required coverage-result columns are missing.
        ValueError: If coverage categories or plotting data are invalid.
    """
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

    metrics = [
        ("mse_test", "MSE", "mse"),
        ("variance_test", "分散", "variance"),
        ("bias_test_sq", "バイアスの二乗", "bias2"),
    ]

    # Distinct markers keep binning strategies distinguishable in grayscale.
    marker_map = {
        "分位点ビニング": "o",
        "最適ビニング": "s",
    }

    for alloc_raw, alloc_jp in allocator_name_map.items():
        current_df = results_df[results_df["標本配分"] == alloc_jp].copy()

        current_df["min_coverage"] = pd.Categorical(
            current_df["min_coverage"], categories=min_coverage_values, ordered=True
        )

        for col_name, label_jp, file_suffix in metrics:
            fig, ax = plt.subplots(figsize=(6, 5))

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

            ax.set_xlabel("被覆率の下限値")
            ax.set_ylabel(label_jp)
            ax.invert_xaxis()
            ax.grid(True, which="major", linestyle="--", alpha=0.7)

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles=handles,
                labels=labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.01),
                ncol=2,
                frameon=True,
                edgecolor="black",
                fancybox=False,
                columnspacing=1.5,
            )

            ax.set_title("")

            out_filename = f"omt_test_{file_suffix}_{alloc_raw}.png"
            plt.savefig(FIG_DIR / out_filename, bbox_inches="tight")
            plt.close(fig)

    print(f"\n✅ 論文用図表（計6枚）を {FIG_DIR} に保存しました。")
