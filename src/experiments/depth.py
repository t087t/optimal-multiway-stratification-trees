"""Rule-length and tree-depth sweep with MIP reduction diagnostics."""

from pathlib import Path

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print
from rich.console import Console

from src.allocator import (
    OptimalAllocator,
    ProportionalAllocator,
    estimate_y_rmse,
)
from src.estimator import (
    CARTEstimator,
    OMTEstimator,
    RandomEstimator,
)

from .config import ExperimentConfig
from .datasets import ExperimentData
from .plots.depth import save as save_plots


def run(data: ExperimentData, config: ExperimentConfig, output_dir: Path):
    """Sweep OMT rule length and CART depth, recording reduction and timing results.

    Args:
        data: Aligned original, encoded, and standardized train/test data views.
        config: Runtime sampling settings, seed, and figure-export options.
        output_dir: Directory in which experiment tables and optional figures are saved.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]: RMSE-reduction, problem-
            reduction, and reduced-MIP timing tables.

    Raises:
        OSError: If the output files cannot be written.
        KeyError: If required result columns are unavailable during figure export.
        ValueError: If figure-export data or plotting parameters are invalid.

    Notes:
        Per-configuration model failures are printed and omitted. The OMT rule-count limit stays
        32 while maximum rule length varies from one through five.
    """
    columns_list_encoded = data.columns_list_encoded
    df_X_train_encoded = data.df_X_train_encoded
    df_X_test_encoded = data.df_X_test_encoded
    X_train_std = data.X_train_std
    X_test_std = data.X_test_std
    y_train = data.y_train
    y_test = data.y_test
    df_X_train_original = data.df_X_train_original
    df_X_test_original = data.df_X_test_original
    SAMPLE_SIZE = config.sample_size
    N_TRIALS = config.n_trials
    RANDOM_STATE = config.random_state
    OUTPUT_DIR = output_dir
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    ALLOCATION_METHODS = [
        ProportionalAllocator(),
        OptimalAllocator(m_value=1),
    ]

    NUM_BINS = 4
    MIN_COVERAGE = 1.0
    MIN_SAMPLES_IN_RULE = int(len(df_X_train_original) * 0.05)

    print(f"MIN_SAMPLES_IN_RULE: {MIN_SAMPLES_IN_RULE}")
    random = RandomEstimator(n_trials=N_TRIALS, sample_size=SAMPLE_SIZE, random_state=RANDOM_STATE)

    console = Console()
    japanize_matplotlib.japanize()
    plt.style.use("seaborn-v0_8-whitegrid")

    MAX_DEPTH_LIST = [1, 2, 3, 4, 5]
    OMT_STRATEGIES = ["quantile", "optbinning"]
    TREE_OUTPUT_DIR = OUTPUT_DIR / "combined_experiment_results"
    TREE_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    console.print("\n--- [ステップ1] ベースラインRMSEの計算 ---", style="bold cyan")

    y_train_true_mean = y_train.mean()
    y_test_true_mean = y_test.mean()

    RMSE_RANDOM_TRAIN = random.predict(X_train_std, y_train, true_mean=y_train_true_mean)
    RMSE_RANDOM_TEST = random.predict(X_test_std, y_test, true_mean=y_test_true_mean)

    all_rrr_results = []
    all_reduction_stats = []
    all_time_results = []

    console.print("\n--- [ステップ2] OMT vs CART 性能 & 計算量検証開始 ---", style="bold cyan")

    for allocator in ALLOCATION_METHODS:
        allocator_name = allocator.__str__()
        console.rule(f"[bold red]Allocator: {allocator_name}[/bold red]")

        for max_depth in MAX_DEPTH_LIST:
            for strategy in OMT_STRATEGIES:
                estimator_name_omt = f"OMT({strategy})"
                console.print(
                    f"\n[bold yellow]実行中: {estimator_name_omt}, 深さ={max_depth}[/bold yellow]"
                )

                omt_params = {
                    "allocator": allocator,
                    "sample_size": SAMPLE_SIZE,
                    "binning_strategy": strategy,
                    "num_bins": NUM_BINS,
                    # Hold the rule-count cap fixed while varying maximum rule length.
                    "max_rules": 2 ** max(MAX_DEPTH_LIST),
                    "max_features_in_rule": max_depth,
                    "min_coverage": MIN_COVERAGE,
                    "min_samples_in_rule": MIN_SAMPLES_IN_RULE,
                    "n_trials": N_TRIALS,
                    "random_state": RANDOM_STATE,
                    "verbose": True,
                    "run_unoptimized_comparison": True,
                }
                if strategy == "optbinning":
                    omt_params.update(
                        {
                            "optb_max_n_prebins": 20,
                            "optb_min_prebin_size": 0.05,
                            "optb_max_n_bins": 20,
                            "optb_min_mean_diff": 0.00,
                            "optb_max_pvalue": 0.05,
                            "optb_gamma": 0.0,
                            "optb_monotonic_trend": "auto",
                        }
                    )

                omt = OMTEstimator(**omt_params)

                try:
                    omt.fit(df_X_train_original, y_train)

                    labels_train_unfiltered = omt.labels_
                    covered_mask_train = labels_train_unfiltered != -1
                    y_train_covered = y_train[covered_mask_train]
                    labels_train_covered = labels_train_unfiltered[covered_mask_train]

                    sample_size_each_cluster = allocator.solve(
                        y_train_covered, labels_train_covered, omt.n_rules_, SAMPLE_SIZE
                    )

                    train_m = estimate_y_rmse(
                        sample_size_each_cluster,
                        y_train_covered,
                        labels_train_covered,
                        y_train_true_mean,
                        N_TRIALS,
                        RANDOM_STATE,
                    )
                    rrr_train = (1 - (train_m / RMSE_RANDOM_TRAIN)) * 100

                    labels_test_unfiltered = omt._assign_labels(df_X_test_original)
                    covered_mask_test = labels_test_unfiltered != -1
                    y_test_covered = y_test[covered_mask_test]
                    labels_test_covered = labels_test_unfiltered[covered_mask_test]

                    test_m = estimate_y_rmse(
                        sample_size_each_cluster,
                        y_test_covered,
                        labels_test_covered,
                        y_test_true_mean,
                        N_TRIALS,
                        RANDOM_STATE,
                    )
                    rrr_test = (1 - (test_m / RMSE_RANDOM_TEST)) * 100

                    all_rrr_results.append(
                        {
                            "allocator_name": allocator_name,
                            "estimator_name": estimator_name_omt,
                            "max_depth": max_depth,
                            "rrr_train": rrr_train,
                            "rrr_test": rrr_test,
                        }
                    )

                    stats = omt.reduction_stats_
                    mip_opt = stats.get("mip_computation_time_sec", np.nan)
                    mip_unopt = stats.get("mip_computation_time_unoptimized_sec", np.nan)

                    all_time_results.append(
                        {
                            "allocator_name": allocator_name,
                            "estimator_name": estimator_name_omt,
                            "max_depth": max_depth,
                            "mip_time_sec": mip_opt,
                        }
                    )

                    all_reduction_stats.append(
                        {
                            "allocator_name": allocator_name,
                            "estimator_name": estimator_name_omt,
                            "max_features": max_depth,
                            "rules_before": stats.get("paths_before", np.nan),
                            "rules_after": stats.get("paths_after_var_reduction", np.nan),
                            "constraints_before": stats.get("instances_before", np.nan),
                            "constraints_after": stats.get(
                                "instances_after_const_reduction", np.nan
                            ),
                            "mip_time_optimized_sec": mip_opt,
                            "mip_time_unoptimized_sec": mip_unopt,
                            "mip_speedup": (mip_unopt / mip_opt)
                            if (mip_opt > 0 and not np.isnan(mip_unopt))
                            else np.nan,
                        }
                    )

                    tree_filename = (
                        TREE_OUTPUT_DIR
                        / f"{allocator_name}_{estimator_name_omt}_depth_{max_depth}.png"
                    )
                    if config.save_trees:
                        omt.save_tree_png(filename=str(tree_filename))

                    console.print(f"  -> OMT完了 (Test RRR: {rrr_test:.2f}%)", style="green")

                except Exception as e:
                    console.print(f"  -> [bold red]OMTエラー: {e}[/bold red]")

            try:
                cart = CARTEstimator(
                    allocator=allocator,
                    sample_size=SAMPLE_SIZE,
                    n_trials=N_TRIALS,
                    max_depth=max_depth,
                    random_state=RANDOM_STATE,
                    min_samples_leaf=MIN_SAMPLES_IN_RULE,
                )
                cart.fit(df_X_train_encoded, y_train)

                c_train_m = cart.predict(df_X_train_encoded, y_train, true_mean=y_train_true_mean)
                c_test_m = cart.predict(df_X_test_encoded, y_test, true_mean=y_test_true_mean)

                rrr_train_c = (1 - (c_train_m / RMSE_RANDOM_TRAIN)) * 100
                rrr_test_c = (1 - (c_test_m / RMSE_RANDOM_TEST)) * 100

                all_rrr_results.append(
                    {
                        "allocator_name": allocator_name,
                        "estimator_name": "CART",
                        "max_depth": max_depth,
                        "rrr_train": rrr_train_c,
                        "rrr_test": rrr_test_c,
                    }
                )

                tree_filename_cart = (
                    TREE_OUTPUT_DIR / f"{allocator_name}_CART_depth_{max_depth}.png"
                )
                if config.save_trees:
                    cart.save_tree_png(
                        feature_names=columns_list_encoded,
                        filename=str(tree_filename_cart),
                    )

                console.print(f"  -> CART完了 (Test RRR: {rrr_test_c:.2f}%)", style="green")

            except Exception as e:
                console.print(f"  -> [bold red]CARTエラー: {e}[/bold red]")

    rrr_df = pd.DataFrame(all_rrr_results)
    reduction_df = pd.DataFrame(all_reduction_stats)
    time_df = pd.DataFrame(all_time_results)

    rrr_df.to_csv(TREE_OUTPUT_DIR / "rrr_comparison_results.csv", index=False)
    reduction_df.to_csv(TREE_OUTPUT_DIR / "computational_reduction_results.csv", index=False)
    time_df.to_csv(TREE_OUTPUT_DIR / "mip_time_vs_depth_results.csv", index=False)
    if config.save_plots:
        save_plots(rrr_df, reduction_df, time_df, MAX_DEPTH_LIST, TREE_OUTPUT_DIR)

    return rrr_df, reduction_df, time_df
