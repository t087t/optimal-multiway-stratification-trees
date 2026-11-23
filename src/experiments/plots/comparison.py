"""Export the original comparison figures without opening windows."""

import japanize_matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save(train_result_df, pred_result_df, OUTPUT_DIR):
    plt.style.use("ggplot")
    TITLE_SIZE, LABEL_SIZE, TICK_SIZE = 20, 15, 12.5
    sfs_result_df = train_result_df[train_result_df["score_history"].notna()]

    for estimator_name, gdf in sfs_result_df.groupby("estimator"):
        # 縦持ちに展開
        gdf["num_features"] = gdf["score_history"].map(lambda x: list(range(1, len(x) + 1)))
        gdf = gdf.explode(["score_history", "num_features"])

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.barplot(
            data=gdf,
            x="num_features",
            y="score_history",
            hue="allocator",
            dodge=True,
            ax=ax,
        )
        ax.set_title(f"{estimator_name}", fontsize=TITLE_SIZE)
        ax.set_xlabel("Number of features", fontsize=LABEL_SIZE)
        ax.set_ylabel("Score", fontsize=LABEL_SIZE)
        ax.legend(
            title="Method",
            fontsize=TICK_SIZE,
        )
        fig.savefig(
            OUTPUT_DIR / f"feature_selection_{estimator_name}.png",
            bbox_inches="tight",
        )
        plt.close(fig)

    japanize_matplotlib.japanize()

    # カラー設定
    COLOR_OPTIMAL = "#6b1314"  # 最適標本配分 (濃い赤)
    COLOR_PROPORTIONAL = "#b87a76"  # 比例配分 (薄い赤)
    COLOR_SINGLE = "#787878"  # CUPEDなど単独の手法 (グレー)

    # フォントサイズ設定
    FONT_SIZE_LABEL = 25
    FONT_SIZE_TICK = 25
    FONT_SIZE_LEGEND = 25
    FONT_SIZE_AXIS_LABEL = 30

    # 表示する手法の順序
    estimator_order = [
        "CART",
        "SFS-KMeans-Var",
        "KMeans-DT",
        "KMeans",
        "OMT(optbinning)",
        "OMT(quantile)",
        "CUPED",
    ]

    # --- 1. データの前処理 (テストデータのみを抽出) ---
    df_test = pred_result_df[~pred_result_df["is_train"]].copy()
    df_test["allocator"] = df_test["allocator"].fillna("N/A").astype(str)

    # estimatorとallocatorで集計
    pivot_df = df_test.pivot_table(
        index="estimator", columns="allocator", values="rrr", aggfunc="mean"
    )

    # --- 2. グラフ描画用データの構築 ---
    y_positions = []
    y_labels = []
    bars_values = []
    bars_y = []
    bars_colors = []

    current_y = 0

    for est_name in estimator_order:
        if est_name not in pivot_df.index:
            continue

        row = pivot_df.loc[est_name]

        # 層化抽出（Optimal/Proportional）の有無を確認
        has_optimal = "Optimal" in row and not pd.isna(row["Optimal"])
        has_proportional = "Proportional" in row and not pd.isna(row["Proportional"])

        # --- パターンA: 層化抽出 (最適 or 比例 が存在する場合) ---
        if has_optimal or has_proportional:
            val_opt = row["Optimal"] if has_optimal else 0
            val_prop = row["Proportional"] if has_proportional else 0

            # 最適配分 (下側)
            bars_values.append(val_opt)
            bars_y.append(current_y)
            bars_colors.append(COLOR_OPTIMAL)

            # 比例配分 (上側)
            bars_values.append(val_prop)
            bars_y.append(current_y + 0.9)  # 0.9ずらして配置
            bars_colors.append(COLOR_PROPORTIONAL)

            # Y軸ラベルの位置 (2本のバーの中間)
            y_positions.append(current_y + 0.45)
            y_labels.append(est_name)

            current_y += 2.5  # 次の手法との間隔

        # --- パターンB: 単独手法 (CUPEDなど) ---
        else:
            val = row["N/A"] if "N/A" in row and not pd.isna(row["N/A"]) else 0

            bars_values.append(val)
            bars_y.append(current_y + 0.4)
            bars_colors.append(COLOR_SINGLE)

            y_positions.append(current_y + 0.4)
            y_labels.append(est_name)

            current_y += 2.0

    # --- 3. 描画の実行 ---
    fig, ax = plt.subplots(figsize=(12, 9))  # 凡例スペース確保のため高さを少し広げました

    ax.barh(bars_y, bars_values, height=0.8, color=bars_colors)

    # 軸・刻みの設定
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="x", labelsize=FONT_SIZE_TICK)
    ax.tick_params(axis="y", pad=5)

    # X軸の表示範囲
    if bars_values:
        limit = max(bars_values) * 1.15
        ax.set_xlim(0, limit)

    # --- A. X軸ラベルを図の横幅の中心に配置 ---
    # ax.set_xlabel の代わりに fig.text を使用
    fig.text(0.45, 0, "RMSE削減率 (%)", ha="center", fontsize=FONT_SIZE_AXIS_LABEL)

    # --- B. 凡例を図の横幅の中心に配置 ---
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_PROPORTIONAL),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_OPTIMAL),
    ]

    # bbox_to_anchor の座標系を fig.transFigure にすることで図全体に対する位置指定が可能
    ax.legend(
        legend_handles,
        ["比例配分", "最適配分"],
        fontsize=FONT_SIZE_LEGEND,
        loc="lower center",
        bbox_to_anchor=(0.45, 1.01),  # 図の上部中央
        ncol=2,
        frameon=True,
        edgecolor="black",
        fancybox=False,
        handleheight=1,  # 凡例のマークの高さを調整
        handlelength=3,  # 凡例のマークの長さを調整
    )

    fig.savefig(OUTPUT_DIR / "rmse_reduction.png", bbox_inches="tight")
    plt.close(fig)
