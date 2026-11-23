"""Original notebook coverage experiment and figures."""

from pathlib import Path

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print
from rich.console import Console

from src.allocator import OptimalAllocator, ProportionalAllocator, estimate_y_var
from src.estimator import (
    OMTEstimator,
)

from .config import ExperimentConfig
from .datasets import ExperimentData
from .plots.coverage import save as save_plots


def run(data: ExperimentData, config: ExperimentConfig, output_dir: Path):
    """Evaluate coverage bounds, retaining the notebook bias/variance definition."""
    y_train = data.y_train
    y_test = data.y_test
    df_X_train_original = data.df_X_train_original
    df_X_test_original = data.df_X_test_original
    SAMPLE_SIZE = config.sample_size
    N_TRIALS = config.n_trials
    RANDOM_STATE = config.random_state
    OUTPUT_DIR = output_dir
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    # 実験方法
    # クラスタリング
    N_CLUSTERS = 5  # クラスタ数

    # 標本配分
    ALLOCATION_METHODS = [
        ProportionalAllocator(),
        OptimalAllocator(m_value=1),
    ]

    # OMT用のパラメータ設定
    NUM_BINS = 4  # 連続値を離散化する際のビンの数
    MAX_FEATURES_IN_RULE = 3  # 1つのルールに含める特徴量の最大数
    MIN_SAMPLES_IN_RULE = int(len(df_X_train_original) * 0.05)

    print(f"MIN_SAMPLES_IN_RULE: {MIN_SAMPLES_IN_RULE}")
    # =============================================================================
    #  OMT：Allocator × Binning × MIN_COVERAGE 実験 + 卒論向け可視化（完全版）
    # =============================================================================

    # =============================================================================
    # --- 初期設定 ---
    # =============================================================================
    console = Console()
    japanize_matplotlib.japanize()
    plt.style.use("seaborn-v0_8-whitegrid")

    # ======================================================================
    # 0. 事前：母集団（train/test）の真の平均を計算（bias定義で使う）
    # ======================================================================
    mu_train_pop = float(y_train.mean())
    mu_test_pop = float(y_test.mean())

    print(f"True mean (TRAIN population): {mu_train_pop:.6f}")
    print(f"True mean (TEST population) : {mu_test_pop:.6f}")

    # --- 1. 実験のセットアップ ---
    print("\n実験のセットアップを開始します。")
    min_coverage_values = [1.0, 0.95, 0.9, 0.85, 0.8]
    binning_strategies = ["quantile", "optbinning"]
    results = []

    # ======================================================================
    # 2. 実験ループ (標本配分方法 × ビニング戦略 × 被覆率)
    # ======================================================================
    for allocator in ALLOCATION_METHODS:
        allocator_name = allocator.__str__()
        console.rule(f"[bold red]Allocator: {allocator_name}[/bold red]")

        for binning_strategy in binning_strategies:
            for min_coverage in min_coverage_values:
                console.rule(
                    f"[bold]Strategy: {binning_strategy}, Min Coverage: {min_coverage}[/bold]"
                )

                omt_params = {
                    "allocator": allocator,
                    "sample_size": SAMPLE_SIZE,
                    "binning_strategy": binning_strategy,
                    "num_bins": NUM_BINS,
                    "max_rules": N_CLUSTERS,
                    "max_features_in_rule": MAX_FEATURES_IN_RULE,
                    "min_coverage": min_coverage,
                    "min_samples_in_rule": MIN_SAMPLES_IN_RULE,
                    "n_trials": N_TRIALS,
                    "random_state": RANDOM_STATE,
                    "verbose": True,
                }

                # strategy が 'optbinning' の場合のみ、詳細パラメータを追加
                if binning_strategy == "optbinning":
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

                    actual_coverage = np.nan
                    if hasattr(omt, "actual_coverage_rate_"):
                        actual_coverage = float(omt.actual_coverage_rate_)

                    result_row = {
                        "allocator": allocator_name,
                        "binning_strategy": binning_strategy,
                        "min_coverage": float(min_coverage),
                        "actual_coverage": actual_coverage,
                    }

                    # ======================================================
                    # 3. TRAIN: covered に限定して bias / variance / mse
                    # ======================================================
                    labels_train_all = omt.labels_
                    covered_mask_train = labels_train_all != -1

                    mse_train = variance_train = bias_train = np.nan
                    sample_size_each_cluster = None

                    if not np.any(covered_mask_train):
                        print("  - TRAIN: N/A (カバーされた訓練サンプルなし)")
                    else:
                        y_train_cov = y_train[covered_mask_train]
                        labels_train_cov = labels_train_all[covered_mask_train]

                        # (A) selection bias（除外によるbias）：母集団平均との差
                        mu_train_cov = float(y_train_cov.mean())
                        bias_train = mu_train_cov - mu_train_pop

                        if omt.n_rules_ > 0:
                            # (B) 配分（covered の strata に対して）
                            sample_size_each_cluster = omt.sample_size_each_cluster

                            # (C) variance（covered 集団で抽出）
                            variance_train = float(
                                estimate_y_var(
                                    sample_size_each_cluster,
                                    y_train_cov,
                                    labels_train_cov,
                                    n_trials=N_TRIALS,
                                    random_state=RANDOM_STATE,
                                )
                            )

                            # (D) mse
                            mse_train = float(bias_train**2 + variance_train)

                            print(
                                f"  - TRAIN: MSE={mse_train:.6f}, "
                                f"Var={variance_train:.6f}, "
                                f"Bias={bias_train:.6f} (Bias^2={bias_train**2:.6f}), "
                                f"mu_cov={mu_train_cov:.6f}"
                            )
                        else:
                            print("  - TRAIN: N/A (有効なルールが見つかりませんでした)")

                    result_row.update(
                        {
                            "mse_train": mse_train,
                            "variance_train": variance_train,
                            "bias_train": bias_train,
                        }
                    )

                    # ======================================================
                    # 4. TEST: same rule で covered を作り bias / variance / mse
                    # ======================================================
                    mse_test = variance_test = bias_test = np.nan

                    if sample_size_each_cluster is None or not np.any(sample_size_each_cluster):
                        print(
                            "  - TEST:  N/A (訓練データからサンプル配分を計算できなかったためスキップ)"
                        )
                    else:
                        labels_test_all = omt._assign_labels(df_X_test_original)
                        covered_mask_test = labels_test_all != -1

                        if not np.any(covered_mask_test):
                            print("  - TEST:  N/A (カバーされたテストサンプルなし)")
                        else:
                            y_test_cov = y_test[covered_mask_test]
                            labels_test_cov = labels_test_all[covered_mask_test]

                            test_cluster_sizes = np.bincount(
                                labels_test_cov, minlength=len(sample_size_each_cluster)
                            )
                            if np.any(sample_size_each_cluster > test_cluster_sizes):
                                raise ValueError(
                                    f"【実験中断】Testデータの層サイズ不足。\n"
                                    f"Alloc: {sample_size_each_cluster}, Actual: {test_cluster_sizes}"
                                )

                            # (A) selection bias（除外によるbias）：母集団平均との差
                            mu_test_cov = float(y_test_cov.mean())
                            bias_test = mu_test_cov - mu_test_pop

                            # (B) variance（test の covered 集団で抽出）
                            variance_test = float(
                                estimate_y_var(
                                    sample_size_each_cluster,
                                    y_test_cov,
                                    labels_test_cov,
                                    n_trials=N_TRIALS,
                                    random_state=RANDOM_STATE,
                                )
                            )

                            # (C) mse
                            mse_test = float(bias_test**2 + variance_test)

                            print(
                                f"  - TEST:  MSE={mse_test:.6f}, "
                                f"Var={variance_test:.6f}, "
                                f"Bias={bias_test:.6f} (Bias^2={bias_test**2:.6f}), "
                                f"mu_cov={mu_test_cov:.6f}"
                            )

                    result_row.update(
                        {
                            "mse_test": mse_test,
                            "variance_test": variance_test,
                            "bias_test": bias_test,
                        }
                    )

                    results.append(result_row)

                except Exception as e:
                    print(f"  - エラー発生: {e}。この組み合わせをスキップします。")
                    results.append(
                        {
                            "allocator": allocator_name,
                            "binning_strategy": binning_strategy,
                            "min_coverage": float(min_coverage),
                            "actual_coverage": np.nan,
                            "mse_train": np.nan,
                            "variance_train": np.nan,
                            "bias_train": np.nan,
                            "mse_test": np.nan,
                            "variance_test": np.nan,
                            "bias_test": np.nan,
                        }
                    )

    # =============================================================================
    # 5. 結果の集計と可視化（卒論・論文投稿向け：個別保存・横並び凡例版）
    # =============================================================================
    print("\n" + "=" * 80 + "\n📊 論文用図表の出力処理を開始します。")

    # 結果の整形
    results_df = pd.DataFrame(results)
    results_df["bias_train_sq"] = results_df["bias_train"] ** 2
    results_df["bias_test_sq"] = results_df["bias_test"] ** 2

    # 表示名（論文用の日本語）
    strategy_name_map = {
        "quantile": "分位点ビニング",
        "optbinning": "最適ビニング",
    }
    allocator_name_map = {
        "Proportional": "比例配分",
        "Optimal": "最適標本配分",
    }

    results_df["ビニング手法"] = (
        results_df["binning_strategy"].map(strategy_name_map).fillna(results_df["binning_strategy"])
    )
    results_df["標本配分"] = (
        results_df["allocator"].map(allocator_name_map).fillna(results_df["allocator"])
    )

    results_df.to_csv(OUTPUT_DIR / "coverage_results.csv", index=False)
    if config.save_plots:
        save_plots(results_df, min_coverage_values, allocator_name_map, OUTPUT_DIR)
    return results_df
