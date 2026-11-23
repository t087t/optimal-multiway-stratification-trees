"""Main method comparison with the original training and evaluation order."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from rich import print
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.allocator import (
    OptimalAllocator,
    ProportionalAllocator,
    estimate_y_rmse,
)
from src.estimator import (
    CARTEstimator,
    CUPEDEstimator,
    KMeansDTEstimator,
    NormalEstimator,
    OMTEstimator,
    RandomEstimator,
    SequentialForwardSelectionVarEstimator,
)

from .config import ExperimentConfig
from .datasets import ExperimentData
from .plots.comparison import save as save_plots


def run(data: ExperimentData, config: ExperimentConfig, output_dir: Path):
    """Fit and evaluate the methods in notebook order; return both result tables."""
    features_list = data.features_list
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
    OUTPUT_DIR = output_dir
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    estimators = build_estimators(data, config)
    random = estimators[0]

    console = Console()

    train_results = []
    for i, estimator in enumerate(estimators, start=1):
        estimator_name = str(estimator)

        console.print(
            Panel.fit(
                f"🎯 [bold cyan]{estimator_name}[/bold cyan]",
                title=f"🧠 Estimator {i}",
                style="blue",
            )
        )

        allocator = getattr(estimator, "allocator", None)
        allocator_name = allocator.__str__() if allocator is not None else "N/A"

        # 学習
        if estimator_name == "KMeans-DT":
            start = time.time()
            estimator.fit(X_train_std, y_train, X_orig=df_X_train_encoded)
            elapsed_time = time.time() - start

        elif "OMT(" in estimator_name:
            X_fit, y_fit = df_X_train_original, y_train
            start = time.time()
            estimator.fit(X_fit, y_fit)
            elapsed_time = time.time() - start

        elif estimator_name == "CART":
            X_fit, y_fit = df_X_train_encoded, y_train
            start = time.time()
            estimator.fit(X_fit, y_fit)
            elapsed_time = time.time() - start

        else:
            X_fit, y_fit = X_train_std, y_train
            start = time.time()
            estimator.fit(X_fit, y_fit)
            elapsed_time = time.time() - start

        # --- 決定木の画像保存 ---
        if config.save_trees and estimator_name == "KMeans-DT":
            tree_filename = (
                OUTPUT_DIR
                / f"decision_tree_{estimator_name}_{allocator_name}_{str(SAMPLE_SIZE)}.png"
            )
            estimator.save_decision_tree(
                feature_names=columns_list_encoded, filename=str(tree_filename)
            )

        elif config.save_trees and estimator_name == "CART":
            tree_filename = (
                OUTPUT_DIR
                / f"decision_tree_{estimator_name}_{allocator_name}_{str(SAMPLE_SIZE)}.png"
            )
            estimator.save_tree_png(feature_names=columns_list_encoded, filename=str(tree_filename))

        elif config.save_trees and "OMT(" in estimator_name:
            tree_filename = (
                OUTPUT_DIR
                / f"decision_tree_{estimator_name}_{allocator_name}_{str(SAMPLE_SIZE)}.png"
            )
            estimator.save_tree_png(filename=str(tree_filename))

        # 結果の整理
        selected_features_index = getattr(estimator, "get_selected_features_index", lambda: None)()

        # estimatorに応じて使用する特徴量名リストを決定する
        if "OMT(" in estimator_name:
            current_feature_names = features_list
        else:
            current_feature_names = columns_list_encoded

        selected_features_name = (
            [current_feature_names[i] for i in selected_features_index]
            if selected_features_index is not None
            else None
        )

        max_correlation = getattr(estimator, "get_max_correlation", lambda: None)()
        score_history = getattr(estimator, "get_score_history", lambda: None)()

        # 結果を表形式で表示
        table = Table(show_header=False, box=None)
        table.add_row("📊 [bold yellow]Allocator:[/bold yellow]", str(allocator_name))
        table.add_row(
            "✅ [bold magenta]Selected Features Index:[/bold magenta]",
            str(selected_features_index),
        )
        table.add_row("📊 [bold yellow]Score History:[/bold yellow]", str(score_history))
        table.add_row("⏱️ [bold cyan]Elapsed Time:[/bold cyan]", f"{elapsed_time:.2f} sec")
        console.print(table)

        # 結果の保存
        train_results.append(
            {
                "allocator": allocator_name,
                "estimator": estimator_name,
                "selected_features": selected_features_index,
                "selected_features_name": selected_features_name,
                "max_correlation": max_correlation,
                "score_history": score_history,
                "elapsed_time": elapsed_time,
            }
        )
        console.print("=" * 60)

    train_result_df = pd.DataFrame(train_results)
    train_result_df.to_csv(OUTPUT_DIR / f"training_result_{str(SAMPLE_SIZE)}.csv", index=False)

    # --- 1. ベースラインとなる単純無作為抽出(Random)のRMSEを計算 ---
    # 訓練データとテストデータの真の平均を先に計算
    y_train_true_mean = y_train.mean()
    y_test_true_mean = y_test.mean()

    # random推定器を使ってベースラインRMSEを計算
    RMSE_RANDOM_TRAIN = random.predict(X_train_std, y_train, true_mean=y_train_true_mean)
    RMSE_RANDOM_TEST = random.predict(X_test_std, y_test, true_mean=y_test_true_mean)

    print(f"訓練データでのRMSE (Random) : {RMSE_RANDOM_TRAIN:.4f}")
    print(f"テストデータでのRMSE (Random) : {RMSE_RANDOM_TEST:.4f}")

    pred_results = []

    for estimator in estimators:
        estimator_name = str(estimator)

        # allocatorをestimatorオブジェクトから直接取得する処理を先に行う
        allocator = getattr(estimator, "allocator", None)
        allocator_name = allocator.__str__() if allocator is not None else "N/A"

        # タイトル表示をestimatorごとに行う
        console.rule(f"[bold cyan]【{estimator_name}({allocator_name})】[/bold cyan]")
        estimator_result = {}

        # --- OMTモデルの処理 ---
        if "OMT(" in estimator_name:
            # 訓練データとテストデータのラベルを取得
            labels_train_unfiltered = estimator.labels_
            labels_test_unfiltered = estimator._assign_labels(df_X_test_original)

            for y_, labels_unfiltered_, random_rmse, true_mean, data_name in [
                (
                    y_train,
                    labels_train_unfiltered,
                    RMSE_RANDOM_TRAIN,
                    y_train_true_mean,
                    "train",
                ),
                (
                    y_test,
                    labels_test_unfiltered,
                    RMSE_RANDOM_TEST,
                    y_test_true_mean,
                    "test",
                ),
            ]:
                covered_mask = labels_unfiltered_ != -1

                if not np.any(covered_mask):
                    rmse, _variance, _bias, rrr = np.nan, np.nan, np.nan, np.nan
                    sample_size_each_cluster = None
                else:
                    y_covered = y_[covered_mask]
                    labels_covered = labels_unfiltered_[covered_mask]

                    # sample_size_each_clusterは訓練データでのみ計算
                    if data_name == "train":
                        sample_size_each_cluster = allocator.solve(
                            y_covered,
                            labels_covered,
                            estimator.n_rules_,
                            estimator.sample_size,
                        )

                    rmse = estimate_y_rmse(
                        sample_size_each_cluster=sample_size_each_cluster,
                        y=y_covered,
                        labels=labels_covered,
                        true_mean=true_mean,
                        n_trials=estimator.n_trials,
                        random_state=estimator.random_state,
                    )

                    rrr = (1 - (rmse / random_rmse)) * 100 if not np.isnan(rmse) else np.nan

                result_data = {
                    "estimator": estimator_name,
                    "allocator": allocator_name,
                    "is_train": data_name == "train",
                    "sample_size_each_cluster": sample_size_each_cluster,
                    "rmse": rmse,
                    "rrr": rrr,
                }
                pred_results.append(result_data)
                estimator_result[data_name] = result_data

        # --- OMT以外のモデルの処理 ---
        else:
            if estimator_name in ["CART", "KMeans-DT"]:
                X_train_to_use, X_test_to_use = df_X_train_encoded, df_X_test_encoded
            else:
                X_train_to_use, X_test_to_use = X_train_std, X_test_std

            for X_, y_, random_rmse, true_mean, data_name in [
                (
                    X_train_to_use,
                    y_train,
                    RMSE_RANDOM_TRAIN,
                    y_train_true_mean,
                    "train",
                ),
                (X_test_to_use, y_test, RMSE_RANDOM_TEST, y_test_true_mean, "test"),
            ]:
                rmse = estimator.predict(X_, y_, true_mean=true_mean)
                sample_size_each_cluster = getattr(estimator, "sample_size_each_cluster", None)
                rrr = (1 - (rmse / random_rmse)) * 100

                result_data = {
                    "estimator": estimator_name,
                    "allocator": allocator_name,
                    "is_train": data_name == "train",
                    "sample_size_each_cluster": sample_size_each_cluster,
                    "rmse": rmse,
                    "rrr": rrr,
                }
                pred_results.append(result_data)
                estimator_result[data_name] = result_data

        # --- 結果テーブルの表示（全モデル共通） ---
        table = Table(title="Results", title_style="bold cyan")
        table.add_column("Metric", justify="left", style="bold yellow")
        table.add_column("Train", justify="right", style="green")
        table.add_column("Test", justify="right", style="red")
        table.add_row(
            "Sample Size Each Cluster",
            str(estimator_result["train"]["sample_size_each_cluster"]),
            str(estimator_result["test"]["sample_size_each_cluster"]),
        )
        table.add_row(
            "RMSE",
            f"{estimator_result['train']['rmse']:.4f}",
            f"{estimator_result['test']['rmse']:.4f}",
        )
        table.add_row(
            "RMSE Reduction Rate",
            f"{estimator_result['train']['rrr']:.2f}%",
            f"{estimator_result['test']['rrr']:.2f}%",
        )
        console.print(table)

    pred_result_df = pd.DataFrame(pred_results)
    pred_result_df.to_csv(OUTPUT_DIR / f"prediction_result_{str(SAMPLE_SIZE)}.csv", index=False)
    if config.save_plots:
        save_plots(train_result_df, pred_result_df, OUTPUT_DIR)
    return train_result_df, pred_result_df


def build_estimators(data: ExperimentData, config: ExperimentConfig):
    """Construct the original comparison methods with unchanged model settings."""
    features_list_cat = data.features_list_cat
    df_X_train_original = data.df_X_train_original
    SAMPLE_SIZE = config.sample_size
    N_TRIALS = config.n_trials
    RANDOM_STATE = config.random_state
    # 実験方法
    # クラスタリング
    CLUSTERING_METHOD = "KMeans"  # "GMM" or "KMeans"
    N_CLUSTERS = 5  # クラスタ数

    # KMeans+DTの葉の数
    MAX_LEAF_NODES = 5

    # 特徴量選択
    # True or No（特徴量数が MAXIMUM_FEATURES_TO_SELECT になるまで選ぶかいなか）
    SELECT_MAXIMUM_FEATURES = False
    MAXIMUM_FEATURES_TO_SELECT = 100  # 選択される最大の特徴量(特徴量選択ありの場合)

    # 標本配分
    ALLOCATION_METHODS = [
        ProportionalAllocator(),
        OptimalAllocator(m_value=1),
    ]

    # OMT用のパラメータ設定
    NUM_BINS = 4  # 連続値を離散化する際のビンの数
    MAX_FEATURES_IN_RULE = 3  # 1つのルールに含める特徴量の最大数
    MIN_COVERAGE = 1.0  # ルールがカバーすべきサンプルの最小割合
    MIN_SAMPLES_IN_RULE = int(len(df_X_train_original) * 0.05)

    print(f"MIN_SAMPLES_IN_RULE: {MIN_SAMPLES_IN_RULE}")
    estimators = []

    # Random
    random = RandomEstimator(
        n_trials=N_TRIALS,
        sample_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    )
    estimators.append(random)

    # CUPED
    cuped = CUPEDEstimator(
        n_trials=N_TRIALS,
        sample_size=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    )
    estimators.append(cuped)

    ##############################
    # 層化抽出手法
    ##############################
    for allocator in ALLOCATION_METHODS:
        # OMTによる層化 (quantile)
        omt_quantile = OMTEstimator(
            allocator=allocator,
            sample_size=SAMPLE_SIZE,
            binning_strategy="quantile",  # 分位数戦略
            num_bins=NUM_BINS,  # 分割数を指定
            max_rules=N_CLUSTERS,
            max_features_in_rule=MAX_FEATURES_IN_RULE,
            categorical_variables=features_list_cat,
            min_coverage=MIN_COVERAGE,
            min_samples_in_rule=MIN_SAMPLES_IN_RULE,
            n_trials=N_TRIALS,
            random_state=RANDOM_STATE,
            verbose=True,
        )
        estimators.append(omt_quantile)

        # OMTによる層化 (optbinning)
        omt_optbinning = OMTEstimator(
            allocator=allocator,
            sample_size=SAMPLE_SIZE,
            binning_strategy="optbinning",
            categorical_variables=features_list_cat,
            max_rules=N_CLUSTERS,
            max_features_in_rule=MAX_FEATURES_IN_RULE,
            min_coverage=MIN_COVERAGE,
            min_samples_in_rule=MIN_SAMPLES_IN_RULE,
            n_trials=N_TRIALS,
            random_state=RANDOM_STATE,
            verbose=True,
            optb_max_n_prebins=20,  # 事前分割の最大数
            optb_min_prebin_size=0.05,  # 各ビンの最小サンプル割合 (5%)
            optb_max_n_bins=20,  # 最終的なビン数の上限（Noneは自動）
            optb_min_mean_diff=0.00,  # 隣接するビン間の平均値の最小差
            optb_max_pvalue=0.05,  # ビンを結合しないための最大p値（統計的有意性の閾値）
            optb_gamma=0.0,  # ビンのサイズを均一化する正則化項
            optb_monotonic_trend="auto",  # 目標値に対する単調性の制約 ("ascending", "descending" など)
        )
        estimators.append(omt_optbinning)

        # KMeans
        normal = NormalEstimator(
            n_clusters=N_CLUSTERS,
            allocator=allocator,
            sample_size=SAMPLE_SIZE,
            n_trials=N_TRIALS,
            clustering_method=CLUSTERING_METHOD,
            random_state=RANDOM_STATE,
        )
        estimators.append(normal)

        # KMeans + 決定木
        kmeans_dt = KMeansDTEstimator(
            n_clusters=N_CLUSTERS,
            allocator=allocator,
            sample_size=SAMPLE_SIZE,
            n_trials=N_TRIALS,
            random_state=RANDOM_STATE,
            max_leaf_nodes=MAX_LEAF_NODES,  # max_depth から変更
        )
        estimators.append(kmeans_dt)

        # クラスタリング + 特徴量選択
        sfs_var = SequentialForwardSelectionVarEstimator(
            maximum_features_to_select=MAXIMUM_FEATURES_TO_SELECT,
            n_clusters=N_CLUSTERS,
            sample_size=SAMPLE_SIZE,
            n_trials=N_TRIALS,
            clustering_method=CLUSTERING_METHOD,
            select_maximum_features=SELECT_MAXIMUM_FEATURES,
            allocator=allocator,
            random_state=RANDOM_STATE,
        )
        estimators.append(sfs_var)

        # CART
        cart = CARTEstimator(
            allocator=allocator,
            sample_size=SAMPLE_SIZE,
            n_trials=N_TRIALS,
            max_leaf_nodes=N_CLUSTERS,  # 葉の数を他の手法のクラスタ数と合わせる
            random_state=RANDOM_STATE,
            min_samples_leaf=MIN_SAMPLES_IN_RULE,
        )
        estimators.append(cart)

    return estimators
