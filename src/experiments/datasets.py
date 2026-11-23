"""Dataset preparation, preserving notebook column order, casts and splits."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig


def _require_columns(frame: pd.DataFrame, columns: dict[str, str], csv_path: Path):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(
            f"Missing columns in {csv_path}: {', '.join(missing)}. See README.md for the input schema."
        )


@dataclass
class ExperimentData:
    features_list: list[str]
    features_list_cat: list[str]
    columns_list_encoded: list[str]
    df_X_train_encoded: pd.DataFrame
    df_X_test_encoded: pd.DataFrame
    X_train_std: np.ndarray
    X_test_std: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    df_X_train_original: pd.DataFrame
    df_X_test_original: pd.DataFrame


def prepare_customer(csv_path: Path, config: ExperimentConfig) -> ExperimentData:
    """Prepare the original and encoded views using the original notebook steps."""
    RANDOM_STATE = config.random_state
    ### 実験データの設定 ###
    TARGET = "購入金額"
    features_list_con = ["年齢", "年収"]  # 連続変数のリスト
    features_list_cat = [
        "性別",
        "学歴",
        "地域",
        "会員ランク",
        "購入頻度",
    ]  # 質的変数のリスト
    features_list = features_list_con + features_list_cat
    [TARGET] + features_list_con + features_list_cat

    # 特徴量名の翻訳マップ
    translation_map = {
        "purchase_amount": "購入金額",
        "age": "年齢",
        "income": "年収",
        "gender": "性別",
        "education": "学歴",
        "region": "地域",
        "loyalty_status": "会員ランク",
        "purchase_frequency": "購入頻度",
    }

    value_translation_map = {
        "gender": {"Male": "男性", "Female": "女性"},
        "education": {
            "HighSchool": "高校",
            "College": "専門",
            "Bachelor": "学士",
            "Masters": "修士",
        },
        "region": {"North": "北", "South": "南", "East": "東", "West": "西"},
        "loyalty_status": {
            "Regular": "レギュラー",
            "Silver": "シルバー",
            "Gold": "ゴールド",
        },
        "purchase_frequency": {"frequent": "高", "occasional": "中", "rare": "低"},
    }

    df_customer = pd.read_csv(csv_path)
    _require_columns(df_customer, translation_map, csv_path)

    for col, mapping in value_translation_map.items():
        if col in df_customer.columns:
            df_customer[col] = df_customer[col].replace(mapping)

    # 3. カラム名を日本語に置換
    df_customer = df_customer.rename(columns=translation_map)

    df_customer_raw = df_customer.copy()

    df_customer["性別"] = df_customer["性別"].map({"男性": 0, "女性": 1})

    # 1-1. 質的変数のリストから「性別」を除外してOne-hotエンコーディングを適用
    features_list_cat_ohe = [f for f in features_list_cat if f != "性別"]
    df_customer_encoded = pd.get_dummies(df_customer, columns=features_list_cat_ohe)

    # 1-2. エンコーディング後の列名リストを再構成
    # 「性別」はそのまま数値カラムとして残っているので、features_list_con（連続変数）と同様に扱います
    binary_feature_names = [
        col
        for col in df_customer_encoded.columns
        if any(col.startswith(f + "_") for f in features_list_cat_ohe)
    ]

    # 性別を明示的に含める形でリストを作成
    columns_list_encoded = features_list_con + ["性別"] + binary_feature_names
    all_columns_encoded = [TARGET] + columns_list_encoded

    # 1-3. 欠損値を除去
    df_customer_clean_encoded = df_customer_encoded[all_columns_encoded].dropna()

    # --- Step 2: 訓練用とテスト用に分割 ---

    # 2-1. 訓練用とテスト用に分割
    df_train_encoded, df_test_encoded = train_test_split(
        df_customer_clean_encoded, test_size=0.5, random_state=RANDOM_STATE
    )

    # 2-2. 【CART / KMeans-DT用】One-hotエンコード済み・"非"標準化データ (DataFrame) を作成
    df_X_train_encoded = df_train_encoded[columns_list_encoded].astype(int)
    df_X_test_encoded = df_test_encoded[columns_list_encoded].astype(int)

    rename_dict = {"性別": "性別_女性"}
    df_X_train_encoded = df_X_train_encoded.rename(columns=rename_dict)
    df_X_test_encoded = df_X_test_encoded.rename(columns=rename_dict)

    columns_list_encoded = [rename_dict.get(c, c) for c in columns_list_encoded]

    # 2-3. 【標準化が必要なモデル用】にNumPy配列に変換して標準化
    X_train_encoded_np = df_X_train_encoded.to_numpy()
    X_test_encoded_np = df_X_test_encoded.to_numpy()
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train_encoded_np)
    X_test_std = sc.transform(X_test_encoded_np)

    # 2-4. y (目的変数) を作成 (これは全モデル共通)
    y_train = df_train_encoded[TARGET].to_numpy()
    y_test = df_test_encoded[TARGET].to_numpy()
    # --- Step 3: 【OMT用】One-hotエンコーディング "前" のデータを作成 ---
    columns_list_original = features_list_con + features_list_cat
    all_columns_original = [TARGET] + columns_list_original
    df_customer_clean_original = df_customer_raw[all_columns_original].dropna()

    # Step 2-1で作成したインデックスを使用して行を抽出する
    df_train_original = df_customer_clean_original.loc[df_train_encoded.index]
    df_test_original = df_customer_clean_original.loc[df_test_encoded.index]

    df_X_train_original = df_train_original[columns_list_original]
    df_X_test_original = df_test_original[columns_list_original]
    return ExperimentData(
        features_list=features_list,
        features_list_cat=features_list_cat,
        columns_list_encoded=columns_list_encoded,
        df_X_train_encoded=df_X_train_encoded,
        df_X_test_encoded=df_X_test_encoded,
        X_train_std=X_train_std,
        X_test_std=X_test_std,
        y_train=y_train,
        y_test=y_test,
        df_X_train_original=df_X_train_original,
        df_X_test_original=df_X_test_original,
    )


def prepare_seoulbike(csv_path: Path, config: ExperimentConfig) -> ExperimentData:
    """Prepare the original and encoded views using the original notebook steps."""
    RANDOM_STATE = config.random_state
    # 実験データの設定
    TARGET = "レンタル自転車数"
    features_list_con = [
        "時間",
        "気温",
        "湿度",
        "風速",
        "視程",
        "露点温度",
        "日射量",
        "降水量",
        "降雪量",
    ]
    features_list_cat = ["季節", "祝日", "稼働日"]
    features_list = features_list_con + features_list_cat
    [TARGET] + features_list_con + features_list_cat

    # 特徴量名の翻訳マップ
    translation_map = {
        "Hour": "時間",
        "Temperature": "気温",
        "Humidity": "湿度",
        "Wind_speed": "風速",
        "Visibility": "視程",
        "Dew_point_temperature": "露点温度",
        "Solar_Radiation": "日射量",
        "Rainfall": "降水量",
        "Snowfall": "降雪量",
        "Seasons": "季節",
        "Holiday": "祝日",
        "Functioning_Day": "稼働日",
        "Rented_Bike_Count": "レンタル自転車数",
    }

    value_translation_map = {
        "Seasons": {"Spring": "春", "Summer": "夏", "Autumn": "秋", "Winter": "冬"},
        "Holiday": {"Holiday": "はい", "No Holiday": "いいえ"},
        "Functioning_Day": {"Yes": "はい", "No": "いいえ"},
    }
    df_Stu = pd.read_csv(csv_path)
    _require_columns(df_Stu, translation_map, csv_path)

    for col, mapping in value_translation_map.items():
        if col in df_Stu.columns:
            df_Stu[col] = df_Stu[col].replace(mapping)

    # 3. カラム名を日本語に置換
    df_Stu = df_Stu.rename(columns=translation_map)

    df_Stu_raw = df_Stu.copy()
    # --- Step 1: まずOne-hotエンコーディングなど通常の前処理を行う ---

    df_Stu["祝日"] = df_Stu["祝日"].map({"はい": 1, "いいえ": 0})
    df_Stu["稼働日"] = df_Stu["稼働日"].map({"はい": 1, "いいえ": 0})

    # 1-1. 質的変数をOne-hotエンコーディング
    features_list_cat_ohe = ["季節"]
    df_Stu_encoded = pd.get_dummies(df_Stu, columns=features_list_cat_ohe)

    # 1-2. One-hotエンコーディング後の列名リストを作成
    binary_feature_names = [col for col in df_Stu_encoded.columns if col.startswith("季節_")]
    columns_list_encoded = features_list_con + ["祝日", "稼働日"] + binary_feature_names
    all_columns_encoded = [TARGET] + columns_list_encoded

    # 1-3. 欠損値を除去
    df_Stu_clean_encoded = df_Stu_encoded[all_columns_encoded].dropna()

    # --- Step 2: 標準化が必要なモデル用のデータを作成 ---

    # 2-1. 訓練用とテスト用に分割
    df_train_encoded, df_test_encoded = train_test_split(
        df_Stu_clean_encoded, test_size=0.5, random_state=RANDOM_STATE
    )

    # 2-2. 【CART / KMeans-DT用】One-hotエンコード済み・"非"標準化データ (DataFrame) を作成
    df_X_train_encoded = df_train_encoded[columns_list_encoded].astype(int)
    df_X_test_encoded = df_test_encoded[columns_list_encoded].astype(int)

    # 2-3. 【標準化が必要なモデル用】にNumPy配列に変換して標準化
    X_train_encoded_np = df_X_train_encoded.to_numpy()
    X_test_encoded_np = df_X_test_encoded.to_numpy()
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train_encoded_np)
    X_test_std = sc.transform(X_test_encoded_np)

    # 2-4. y (目的変数) を作成 (これは全モデル共通)
    y_train = df_train_encoded[TARGET].to_numpy()
    y_test = df_test_encoded[TARGET].to_numpy()

    # 【OMT用】One-hotエンコーディング "前" のデータを作成 ---
    columns_list_original = features_list_con + features_list_cat
    all_columns_original = [TARGET] + columns_list_original
    df_Stu_clean_original = df_Stu_raw[all_columns_original].dropna()

    # Step 2-1で作成したインデックスを使用して行を抽出する
    df_train_original = df_Stu_clean_original.loc[df_train_encoded.index]
    df_test_original = df_Stu_clean_original.loc[df_test_encoded.index]

    df_X_train_original = df_train_original[columns_list_original]
    df_X_test_original = df_test_original[columns_list_original]
    return ExperimentData(
        features_list=features_list,
        features_list_cat=features_list_cat,
        columns_list_encoded=columns_list_encoded,
        df_X_train_encoded=df_X_train_encoded,
        df_X_test_encoded=df_X_test_encoded,
        X_train_std=X_train_std,
        X_test_std=X_test_std,
        y_train=y_train,
        y_test=y_test,
        df_X_train_original=df_X_train_original,
        df_X_test_original=df_X_test_original,
    )
