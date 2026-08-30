# omt.py

# -*- coding: utf-8 -*-

import gc
import itertools
import time
from collections import defaultdict
from typing import Any, Dict, List, Literal, Tuple, Union

import graphviz
import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB
from numpy.typing import NDArray
from optbinning import BinningProcess, ContinuousOptimalBinning
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted

from src.allocator import AllocatorProtocol, OptimalAllocator, ProportionalAllocator, estimate_y_rmse

MASK_BATCH_SIZE = 4096
MIP_TIME_LIMIT_SECONDS = 3600
MIP_THREADS = 1


class OMTEstimator(RegressorMixin, BaseEstimator):
    """
    数理最適化を用いた最適な層化ルール学習用Estimator。
    """

    def __init__(
        self,
        allocator: "AllocatorProtocol",
        sample_size: int,
        binning_strategy: Literal['quantile', 'optbinning'] = 'quantile',
        categorical_variables: List[str] = None,
        num_bins: int = 4,
        max_features_in_rule: int = 3,
        max_rules: int = 6,
        min_coverage: float = 1.0,
        min_samples_in_rule: int = 10,
        n_trials: int = 10000,
        verbose: bool = True,
        random_state: int = 0,
        run_unoptimized_comparison: bool = False,
        optb_max_n_prebins: int = 20,
        optb_min_prebin_size: float = 0.05,
        optb_max_n_bins: int | None = None,
        optb_min_mean_diff: float = 0.0,
        optb_max_pvalue: float | None = None,
        optb_gamma: float = 0.0,
        optb_monotonic_trend: str | None = None,
    ):
        self.allocator = allocator
        self.sample_size = sample_size
        self.binning_strategy = binning_strategy
        self.categorical_variables = categorical_variables
        self.num_bins = num_bins
        self.max_features_in_rule = max_features_in_rule
        self.max_rules = max_rules
        self.min_coverage = min_coverage
        self.min_samples_in_rule = min_samples_in_rule
        self.n_trials = n_trials
        self.verbose = verbose
        self.random_state = random_state
        self.run_unoptimized_comparison = run_unoptimized_comparison
        self.reduction_stats_ = {}
        self.optb_max_n_prebins = optb_max_n_prebins
        self.optb_min_prebin_size = optb_min_prebin_size
        self.optb_max_n_bins = optb_max_n_bins
        self.optb_min_mean_diff = optb_min_mean_diff
        self.optb_max_pvalue = optb_max_pvalue
        self.optb_gamma = optb_gamma
        self.optb_monotonic_trend = optb_monotonic_trend

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> "OMTEstimator":
        """
        データから最適な層化ルールを学習する。
        """

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f"feature_{i}" for i in range(X.shape[1])]

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()

            overall_start_time = time.time()

            X_df, y_s = self._validate_and_convert_input(X, y)
            if self.binning_strategy == "optbinning":
                self._fit_optbinning_process(X_df, y_s)

            n_samples = X_df.shape[0]

            if self.verbose:
                print("\n[ステップ1/4] ルール候補の生成")
            all_rules = self._generate_rules(X_df, y_s)

            self.reduction_stats_['instances_before'] = n_samples
            self.reduction_stats_['paths_before'] = len(all_rules)

            if self.run_unoptimized_comparison:
                if self.verbose:
                    print(f"\n比較のため、削減前の全ルール ({len(all_rules)}個) で縮小なしMIPを実行")
                self.reduction_stats_["mip_computation_time_unoptimized_sec"] = (
                    self._solve_mip_unoptimized(all_rules, n_samples, env)
                )

            if self.verbose:
                print("\n[ステップ2/4] 決定変数の削減")
            unique_rules = self._reduce_decision_variables(all_rules)
            self.reduction_stats_['paths_after_var_reduction'] = len(unique_rules)

            del all_rules
            gc.collect()

            if self.verbose:
                print("\n[ステップ3/4] 最適なルールの組み合わせ発見 (MIP求解)")
            selected_rules = self._solve_mip(unique_rules, n_samples, env)
            if not selected_rules:
                raise RuntimeError("OMTモデルで有効なルールの組み合わせが見つからなかった")

            self.selected_rules_ = selected_rules
            self.n_rules_ = len(self.selected_rules_)
            self._print_selected_rules()

            if self.verbose:
                print("\n[ステップ4/4] 学習結果の確定と保存")

            start_time = time.time()
            labels = self._assign_labels(X_df)
            self.labels_ = labels
            end_time = time.time()
            if self.verbose:
                print(f"  - ラベル割り当て完了 (所要時間: {end_time - start_time:.4f}秒)")

            covered_mask = (self.labels_ != -1)
            if np.any(covered_mask):
                y_covered = y[covered_mask]
                labels_covered = self.labels_[covered_mask]

                start_time = time.time()
                self.sample_size_each_cluster = self.allocator.solve(
                    y_covered, labels_covered, self.n_rules_, self.sample_size
                )
                end_time = time.time()
                if self.verbose:
                    print(f"  - サンプルサイズ割り当て完了 (所要時間: {end_time - start_time:.4f}秒)")
            else:
                if self.verbose:
                    print("  - 警告: カバーされたサンプルがないため、サンプルサイズ割り当てをスキップ")
                self.sample_size_each_cluster = np.zeros(self.n_rules_, dtype=int)

            self.is_fitted_ = True

            overall_end_time = time.time()
            if self.verbose:
                print(f"fitメソッド全体の所要時間: {overall_end_time - overall_start_time:.2f}秒")

            return self

    def _cumulative_binning(self, series: pd.Series, y: pd.Series) -> List[Tuple[float, float]]:
        """選択された戦略に基づき、連続特徴量のすべての可能な区間を生成する。"""
        quantiles = []

        if self.binning_strategy == 'quantile':
            s = series.dropna().to_numpy()
            uniq = np.sort(np.unique(s))
            if len(uniq) <= self.num_bins:
                quantiles = uniq[1:-1]
            else:
                bins = pd.qcut(series, q=self.num_bins, retbins=True, duplicates='drop')[1]
                quantiles = bins[1:-1]
            if self.verbose:
                print(f"  - 特徴量 '{series.name}': 分位数ビニング (k={self.num_bins})")

        elif self.binning_strategy == "optbinning":
            splits = np.array(self.optb_splits_.get(series.name, []), dtype=float)
            splits = np.sort(np.unique(splits[~np.isnan(splits)]))
            quantiles = splits
            if self.verbose:
                print(f"  - 特徴量 '{series.name}': 最適ビニング (max_n_bins={self.optb_max_n_bins})")

        else:
            raise ValueError(f"未知のビニング戦略: {self.binning_strategy}")

        thresholds = np.concatenate(([-np.inf], np.sort(np.unique(quantiles)), [np.inf]))
        if self.verbose:
            threshold_str = np.array2string(np.round(thresholds, 2), formatter={'float_kind': lambda x: f'{x:.2f}'})
            print(f"    - 閾値: {threshold_str}")

        if len(thresholds) < 2:
            return []

        intervals = [
            pair for pair in itertools.combinations(thresholds, 2)
            if not (pair[0] == -np.inf and pair[1] == np.inf)
        ]

        if self.verbose:
            print(f"    - 閾値の組み合わせから {len(intervals)} 個のルール候補を生成")

        return intervals

    def _generate_rules(self, X: pd.DataFrame, y: pd.Series) -> List[Dict[str, Any]]:
        if self.verbose:
            print("\nルール生成を開始")
            print(f"ルールに含まれる特徴量の最大数: {self.max_features_in_rule}")
            print(f"ルールの最小サンプル数(枝刈り閾値): {self.min_samples_in_rule}")

        start_time = time.time()

        if self.binning_strategy == "optbinning":
            if hasattr(self, "optb_selected_variables_"):
                X = X[self.optb_selected_variables_].copy()
            else:
                X = X[list(self.optb_splits_.keys())].copy()

            if X.shape[1] == 0:
                raise RuntimeError("optbinningで選択された特徴量なし selection_criteriaの緩和が必要")

            if self.verbose:
                print(f"  - optbinning: selected variables only -> {X.columns.tolist()}")

        n_samples = X.shape[0]
        y_values = y.to_numpy()
        nbytes = (n_samples + 7) // 8
        feature_types, X_converted = self._infer_feature_types(X)
        feature_names = sorted(X.columns.tolist(), key=lambda c: X_converted[c].nunique(dropna=True))
        self.feature_order_ = feature_names

        atomic_conditions: List[List[Dict[str, Any]]] = []
        for col in feature_names:
            if self.binning_strategy == "optbinning" and feature_types[col] == "categorical":
                atomic_conditions.append(self._build_optbinning_categorical_conditions(col, X_converted[col]))
                continue

            if feature_types[col] == "categorical":
                atomic_conditions.append(self._build_singleton_categorical_conditions(col, X_converted[col]))
                continue

            atomic_conditions.append(self._build_numeric_conditions(col, X[col], y))

        rules: List[Dict[str, Any]] = []
        initial_mask = np.ones(n_samples, dtype=bool)

        def _build_rules_recursively(feat_idx_start: int, current_features: Dict[str, Any], current_mask: np.ndarray):
            for i in range(feat_idx_start, len(feature_names)):
                if i >= len(atomic_conditions) or len(atomic_conditions[i]) == 0:
                    continue

                for node in atomic_conditions[i]:
                    new_mask = current_mask & node["mask"]
                    num_samples_in_rule = int(new_mask.sum())

                    if num_samples_in_rule < self.min_samples_in_rule:
                        continue

                    new_features = current_features.copy()
                    new_features[node["feature"]] = node["condition"]

                    y_subset = y_values[new_mask]
                    if len(y_subset) > 1:
                        variance_j = float(np.nanvar(y_subset, ddof=1))
                    else:
                        variance_j = 0.0
                    std_j = float(np.sqrt(variance_j)) if variance_j > 0 else 0.0

                    mask_packed = self._pack_mask(new_mask)
                    if len(mask_packed) != nbytes:
                        mask_packed = mask_packed + (b"\x00" * (nbytes - len(mask_packed)))

                    rules.append(
                        {
                            "features": new_features,
                            "mask_packed": mask_packed,
                            "train_samples": num_samples_in_rule,
                            "N_j_std_j": num_samples_in_rule * std_j,
                            "N_j_variance_j": num_samples_in_rule * variance_j,
                            "std_j": std_j,
                        }
                    )

                    if len(new_features) < self.max_features_in_rule:
                        _build_rules_recursively(i + 1, new_features, new_mask)

        _build_rules_recursively(0, {}, initial_mask)

        if self.verbose:
            print(f"生成されたルール数: {len(rules)}個 所要時間: {time.time() - start_time:.2f}秒")

        return rules

    @staticmethod
    def _pack_mask(mask_bool: np.ndarray) -> bytes:
        return np.packbits(mask_bool.astype(np.uint8, copy=False), bitorder="little").tobytes()

    def _infer_feature_types(self, X: pd.DataFrame) -> tuple[Dict[str, str], pd.DataFrame]:
        feature_types: Dict[str, str] = {}
        X_converted = X.copy()

        for col in X.columns:
            is_categorical = (
                col in self.categorical_variables
                if self.categorical_variables is not None
                else not pd.api.types.is_numeric_dtype(X[col])
            )
            feature_types[col] = "categorical" if is_categorical else "numeric"
            if is_categorical:
                X_converted[col] = X_converted[col].astype(str)

        return feature_types, X_converted

    def _make_categorical_condition(self, feature: str, categories: set[str], values: np.ndarray) -> Dict[str, Any]:
        ordered_categories = tuple(sorted(categories))
        return {
            "feature": feature,
            "condition": ("in", ordered_categories),
            "type": "categorical",
            "mask": np.isin(values, ordered_categories),
        }

    def _build_optbinning_categorical_conditions(self, feature: str, series: pd.Series) -> List[Dict[str, Any]]:
        values = series.to_numpy()
        all_categories = set(pd.Series(values).dropna().unique().tolist())
        raw_groups = self.optb_splits_.get(feature, []) if hasattr(self, "optb_splits_") else []

        if len(all_categories) <= 1 or not raw_groups:
            return []

        conditions: List[Dict[str, Any]] = []
        seen: set[frozenset[str]] = set()

        for group in raw_groups:
            group_set = set(map(str, list(group)))
            for candidate in (group_set, all_categories - group_set):
                if not candidate or candidate == all_categories:
                    continue
                key = frozenset(candidate)
                if key in seen:
                    continue
                seen.add(key)
                conditions.append(self._make_categorical_condition(feature, candidate, values))

        return conditions

    def _build_singleton_categorical_conditions(self, feature: str, series: pd.Series) -> List[Dict[str, Any]]:
        values = series.to_numpy()
        categories = pd.Series(values).dropna().unique().tolist()
        return [
            self._make_categorical_condition(feature, {str(category)}, values)
            for category in categories
        ]

    def _build_numeric_conditions(self, feature: str, series: pd.Series, y: pd.Series) -> List[Dict[str, Any]]:
        values = series.to_numpy().astype(float)
        conditions: List[Dict[str, Any]] = []

        for lower, upper in self._cumulative_binning(series, y):
            conditions.append(
                {
                    "feature": feature,
                    "condition": (lower, upper),
                    "type": "numeric",
                    "mask": (values > lower) & (values <= upper),
                }
            )

        return conditions

    def _assign_labels(self, X: pd.DataFrame) -> NDArray[np.int_]:
        """[in演算子対応] ラベル割り当て"""
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1, dtype=int)

        for i, rule in enumerate(self.selected_rules_):
            mask = np.ones(n_samples, dtype=bool)
            for feature, condition in rule["features"].items():
                if isinstance(condition, tuple) and len(condition) == 2:
                    op, val = condition
                    if op == "in":
                        mask &= X[feature].astype(str).isin(val).to_numpy()
                    elif op in ("==", "!="):
                        col = X[feature].astype(str).to_numpy()
                        mask &= (col == str(val)) if op == "==" else (col != str(val))
                    else:
                        lower, upper = condition
                        col = X[feature].to_numpy().astype(float)
                        mask &= (col > lower) & (col <= upper)
                else:
                    mask &= (X[feature].astype(str) == str(condition)).to_numpy()
            labels[mask] = i
        return labels

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

        print("\n" + "="*20 + " 最適化により選択されたパス " + "="*20)
        rules = self.selected_rules_
        for i, rule in enumerate(rules):
            num_samples = rule.get('train_samples', 'N/A')
            std_j = rule.get('std_j', 'N/A')
            std_str = f"{std_j:.3f}" if isinstance(std_j, (int, float)) else str(std_j)

            print(f"\n[層 {i+1}]")
            print(fr"  統計量: 事例数 = {num_samples}, 標準偏差 $\sigma$ = {std_str}")
            print(f"  条件:")
            for feature, condition in rule["features"].items():
                formatted_cond = self._format_condition(feature, condition)
                print(f"    - {formatted_cond}")

    def predict(self, X: NDArray[np.float64], y: NDArray[np.float64], true_mean: float) -> float:
        """学習した層化戦略に基づき、処置群の標本RMSEを推定する。"""
        check_is_fitted(self, "is_fitted_")
        X_df, y_s = self._validate_and_convert_input(X, y)

        labels = self._assign_labels(X_df)

        covered_mask = (labels != -1)

        if not np.any(covered_mask):
            print("警告: 予測対象データにカバーされたサンプルなし")
            return float("nan")

        y_covered = y_s.to_numpy()[covered_mask]
        labels_covered = labels[covered_mask]

        current_cluster_sizes = np.bincount(labels_covered, minlength=len(self.sample_size_each_cluster))

        if np.any(self.sample_size_each_cluster > current_cluster_sizes):
            raise ValueError(
                "【実験中断】テストデータの層内サンプル数が、学習時に決定した配分数を下回った\n"
                f"  - Allocation (n_h): {self.sample_size_each_cluster}\n"
                f"  - Actual Test Size (N_h): {current_cluster_sizes}\n"
                "標本数の不足による実験停止"
            )

        rmse = estimate_y_rmse(
            sample_size_each_cluster=self.sample_size_each_cluster,
            y=y_covered,
            labels=labels_covered,
            true_mean=true_mean,
            n_trials=self.n_trials,
            random_state=self.random_state,
        )
        return rmse

    def _validate_and_convert_input(self, X, y) -> tuple[pd.DataFrame, pd.Series]:
        """入力を検証し、pandas形式に変換"""
        y_checked = check_array(y, ensure_2d=False, dtype="numeric")

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_checked = check_array(X, ensure_2d=True, dtype=None, force_all_finite=False)
            X_df = pd.DataFrame(X_checked, columns=self.feature_names_in_)

        if hasattr(self, 'feature_names_in_') and self.feature_names_in_ != X_df.columns.tolist():
            raise ValueError("fit時とpredict/fit時で特徴量名が異なる")

        y_s = pd.Series(y_checked, name="target", index=X_df.index)
        return X_df, y_s

    def _fit_optbinning_process(self, X: pd.DataFrame, y: pd.Series) -> None:
        variable_names = X.columns.tolist()

        cat_vars = set(self.categorical_variables or [])
        if self.categorical_variables is None:
            for col in variable_names:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    cat_vars.add(col)

        dict_optb = {}
        for col in variable_names:
            dtype = "categorical" if col in cat_vars else "numerical"
            optb = ContinuousOptimalBinning(
                name=col,
                dtype=dtype,
                max_n_prebins=self.optb_max_n_prebins,
                min_prebin_size=self.optb_min_prebin_size,
                max_n_bins=self.optb_max_n_bins,
                min_mean_diff=self.optb_min_mean_diff,
                max_pvalue=self.optb_max_pvalue,
                gamma=self.optb_gamma,
                monotonic_trend=(self.optb_monotonic_trend if self.optb_monotonic_trend is not None else "auto"),
            )
            optb.fit(X[col].to_numpy(), y.to_numpy())
            dict_optb[col] = optb

        self.binning_process_ = BinningProcess(variable_names=variable_names, verbose=False, selection_criteria={
                                               "quality_score": {"min": 0.01, "strategy": "highest", "top": 3}})
        self.binning_process_.fit_from_dict(dict_optb)

        summary_df = self.binning_process_.summary()
        selected_vars = summary_df.loc[summary_df["selected"] == True, "name"].tolist()

        self.optb_selected_variables_ = selected_vars
        self.optb_splits_ = {col: dict_optb[col].splits for col in selected_vars}
        self.optb_is_categorical_ = {col: (col in cat_vars) for col in selected_vars}

        if self.verbose:
            dropped = [c for c in variable_names if c not in selected_vars]
            print(f"\n[OptBinning] Selected variables ({len(selected_vars)}): {selected_vars}")
            print(f"[OptBinning] Dropped variables ({len(dropped)}): {dropped}")

    def _reduce_decision_variables(self, rules: List[Dict]) -> List[Dict]:
        """重複するルールを集約（packed mask版）"""
        if not rules:
            return []

        if self.verbose:
            print(f"\n決定変数の削減を開始 (元のルール数: {len(rules)}個)")
        start_time = time.time()

        sample_set_to_rules: Dict[bytes, List[Dict]] = {}

        for rule in rules:
            mask_key = rule["mask_packed"]
            sample_set_to_rules.setdefault(mask_key, []).append(rule)

        unique_rules: List[Dict] = []
        for _, grouped_rules in sample_set_to_rules.items():
            if not grouped_rules:
                continue
            best_rule = min(grouped_rules, key=lambda r: len(r['features']))
            unique_rules.append(best_rule)

        if self.verbose:
            end_time = time.time()
            print(f"削減後のユニークなルール数: {len(unique_rules)}個 所要時間: {end_time - start_time:.2f}秒")

        return unique_rules

    def _solve_mip(self, rules: List[Dict], num_samples: int, env: gp.Env) -> Union[List[Dict], None]:
        """MIPソルバーで最適なルールを選択"""
        if self.verbose:
            print(f"\nMIPソルバーに投入するルール数: {len(rules)}個")
            print(f"制約: ルール数 <= {self.max_rules}, カバレッジ >= {self.min_coverage:.0%}")
            print(f"目的関数: {self.allocator.__class__.__name__} に基づく標本平均の分散の最小化")
            print(f"  - Allocator: {self.allocator.__class__.__name__}")
            print(f"  - 総サンプル数 (N): {num_samples}")
            print(f"  - 総標本サイズ (n): {self.sample_size}")
            print(f"  - 最小被覆率 (ρ): {self.min_coverage}")

        start_time_grouping = time.time()

        if not rules:
            return None

        nbytes_mask = len(rules[0]["mask_packed"])
        num_rules = len(rules)
        nbytes_key = (num_rules + 7) // 8
        keys = np.zeros((num_samples, nbytes_key), dtype=np.uint8)

        # ルールごとの被覆ビット列をバッチ展開し、同じ被覆候補を持つサンプルを集約する。
        mask_packed_mat = np.empty((num_rules, nbytes_mask), dtype=np.uint8)
        for j, r in enumerate(rules):
            mask_packed_mat[j, :] = np.frombuffer(r["mask_packed"], dtype=np.uint8, count=nbytes_mask)

        for start in range(0, num_rules, MASK_BATCH_SIZE):
            end = min(start + MASK_BATCH_SIZE, num_rules)
            block = mask_packed_mat[start:end, :]

            block_bits = np.unpackbits(block, axis=1, bitorder="little")
            block_bits = block_bits[:, :num_samples].astype(bool, copy=False)

            for local_j in range(end - start):
                j = start + local_j
                mask = block_bits[local_j]
                if not mask.any():
                    continue

                byte = j >> 3
                bit = j & 7
                keys[mask, byte] |= (1 << bit)

            del block_bits

        zero_key = b"\x00" * nbytes_key
        groups_bytes: Dict[bytes, List[int]] = defaultdict(list)
        for i in range(num_samples):
            b = keys[i].tobytes()
            if b == zero_key:
                continue
            groups_bytes[b].append(i)

        self.reduction_stats_['instances_after_const_reduction'] = len(groups_bytes)

        self.actual_covered_samples_ = 0
        self.actual_coverage_rate_ = 0.0

        if self.verbose:
            end_time_grouping = time.time()
            print(f"サンプルを {len(groups_bytes)} 個のグループに集約 (所要時間: {end_time_grouping - start_time_grouping:.2f}秒)")
            print("\nMIPの最適化を開始")

        def bytes_to_rule_tuple(b: bytes) -> Tuple[int, ...]:
            arr = np.frombuffer(b, dtype=np.uint8)
            bits = np.unpackbits(arr, bitorder="little")[:num_rules]
            idx = np.flatnonzero(bits)
            return tuple(idx.tolist())

        groups: Dict[Tuple[int, ...], List[int]] = {}
        for b, idx_list in groups_bytes.items():
            groups[bytes_to_rule_tuple(b)] = idx_list

        del keys, groups_bytes, mask_packed_mat

        with gp.Model("OMT_SetPacking_Optimized", env=env) as model:
            model.setParam("TimeLimit", MIP_TIME_LIMIT_SECONDS)
            model.setParam("Seed", self.random_state)
            model.setParam("Threads", MIP_THREADS)

            z = model.addVars(len(rules), vtype=GRB.BINARY, name="z")

            n = self.sample_size

            sum_N_j_variance_j_z_j = gp.quicksum(rules[j]["N_j_variance_j"] * z[j] for j in range(len(rules)))
            variance_term = None

            if isinstance(self.allocator, OptimalAllocator):
                sum_N_j_std_j_z_j_expr = gp.quicksum(rules[j]["N_j_std_j"] * z[j] for j in range(len(rules)))
                s = model.addVar(lb=0.0, name="aux_sum_std")
                model.addConstr(s == sum_N_j_std_j_z_j_expr, name="aux_constr_std")

                variance_term_1 = (1.0 / n) * (s * s)
                variance_term_2 = sum_N_j_variance_j_z_j
                variance_term = variance_term_1 - variance_term_2

            elif isinstance(self.allocator, ProportionalAllocator):
                variance_term = sum_N_j_variance_j_z_j

            model.setObjective(variance_term, GRB.MINIMIZE)

            for rule_tuple in groups.keys():
                model.addConstr(gp.quicksum(z[j] for j in rule_tuple) <= 1, name="packing_group")

            model.addConstr(gp.quicksum(z) <= self.max_rules, name="max_rules")

            covered_samples_expr = gp.quicksum(
                len(samples_in_group) * gp.quicksum(z[j] for j in rule_tuple)
                for rule_tuple, samples_in_group in groups.items()
            )
            model.addConstr(covered_samples_expr >= self.min_coverage * num_samples, name="coverage")

            model.optimize()

            try:
                status_code = model.Status
            except Exception:
                status_code = None

            status_map = {
                GRB.OPTIMAL: "OPTIMAL",
                GRB.INFEASIBLE: "INFEASIBLE",
                GRB.UNBOUNDED: "UNBOUNDED",
                GRB.INF_OR_UNBD: "INF_OR_UNBD",
                GRB.TIME_LIMIT: "TIME_LIMIT",
                GRB.SUBOPTIMAL: "SUBOPTIMAL",
                GRB.INTERRUPTED: "INTERRUPTED",
            }
            status_name = status_map.get(status_code, str(status_code))

            mip_gap = None
            obj_val = None
            obj_bound = None
            sol_count = None

            try:
                sol_count = model.SolCount
            except Exception:
                sol_count = None

            try:
                if sol_count and sol_count > 0:
                    obj_val = model.ObjVal
            except Exception:
                obj_val = None

            try:
                obj_bound = model.ObjBound
            except Exception:
                obj_bound = None

            try:
                if sol_count and sol_count > 0:
                    mip_gap = model.MIPGap
            except Exception:
                mip_gap = None

            self.reduction_stats_['mip_computation_time_sec'] = model.Runtime
            self.reduction_stats_["mip_status"] = status_name
            self.reduction_stats_["mip_sol_count"] = sol_count
            self.reduction_stats_["mip_obj_val"] = obj_val
            self.reduction_stats_["mip_obj_bound"] = obj_bound
            self.reduction_stats_["mip_gap"] = mip_gap

            if self.verbose:
                print("\n[MIP DIAGNOSTICS]")
                print(f"  - Status      : {status_name} ({status_code})")
                print(f"  - Runtime (s) : {getattr(model, 'Runtime', None)}")
                print(f"  - SolCount    : {sol_count}")
                print(f"  - ObjVal      : {obj_val}")
                print(f"  - ObjBound    : {obj_bound}")
                print(f"  - MIPGap      : {mip_gap}")

            if model.SolCount > 0:
                selected_indices = [j for j, v in z.items() if v.X > 0.5]
                selected_set = set(selected_indices)

                actual_covered_samples = 0
                for rule_tuple, samples_in_group in groups.items():
                    if any(j in selected_set for j in rule_tuple):
                        actual_covered_samples += len(samples_in_group)

                actual_coverage_rate = (actual_covered_samples / num_samples) if num_samples > 0 else 0.0
                self.actual_covered_samples_ = actual_covered_samples
                self.actual_coverage_rate_ = actual_coverage_rate

                if self.verbose:
                    status_text = "最適解"
                    if model.Status == GRB.TIME_LIMIT:
                        status_text = "準最適解 (時間上限)"
                    elif model.Status == GRB.SUBOPTIMAL:
                        status_text = "準最適解"

                    print(f"\nMIPの最適化が完了 ({status_text})")
                    print(f"  - 選択されたルール数: {len(selected_indices)} (上限: {self.max_rules})")
                    print(
                        f"  - カバーされたサンプル数: {int(actual_covered_samples)} / {num_samples} ({actual_coverage_rate:.2%})")
                    print(f"  - (制約上の下限: {self.min_coverage:.2%})")

                return [rules[j] for j in selected_indices]

            else:
                if self.verbose:
                    print("\n最適化は完了したが、有効な解は見つからなかった")
                    if model.Status == GRB.INFEASIBLE:
                        print("  - ステータス: 実行不可能 (Infeasible) 制約が厳しすぎる可能性あり")
                    elif model.Status == GRB.TIME_LIMIT:
                        print("  - ステータス: 時間上限 (実行可能な解が見つかる前に終了)")
                    elif model.Status == GRB.UNBOUNDED:
                        print("  - ステータス: 非有界 (Unbounded) 目的関数の定義の確認が必要")
                    else:
                        print(f"  - ステータスコード: {model.Status}")
                return None

    def _solve_mip_unoptimized(self, rules: List[Dict], num_samples: int, env: gp.Env) -> float:
        """制約削減なしでMIPを解き、計算時間比較用のRuntimeだけ返す。"""
        if not rules:
            return np.nan

        if self.verbose:
            print(f"\n[縮小なしMIP] ルール数: {len(rules)}")

        try:
            cover_lists = self._build_cover_lists_from_packed_masks(rules, num_samples)

            with gp.Model("OMT_SetPacking_Unoptimized", env=env) as model:
                model.setParam("TimeLimit", MIP_TIME_LIMIT_SECONDS)
                model.setParam("Seed", self.random_state)
                model.setParam("Threads", MIP_THREADS)

                z = model.addVars(len(rules), vtype=GRB.BINARY, name="z")
                model.setObjective(self._build_variance_objective(model, z, rules), GRB.MINIMIZE)

                for i, idx in enumerate(cover_lists):
                    if idx.size:
                        model.addConstr(gp.quicksum(z[int(j)] for j in idx) <= 1, name=f"packing_i[{i}]")

                model.addConstr(gp.quicksum(z) <= self.max_rules, name="max_rules")
                covered_samples_expr = gp.quicksum(
                    gp.quicksum(z[int(j)] for j in idx) for idx in cover_lists if idx.size
                )
                model.addConstr(covered_samples_expr >= self.min_coverage * num_samples, name="coverage")

                model.optimize()

                if self.verbose:
                    status = "解あり" if model.SolCount > 0 else f"解なし(status={model.Status})"
                    print(f"  - 縮小なしMIP: {status}, Runtime={model.Runtime:.2f}秒")

                return float(model.Runtime)

        except (gp.GurobiError, MemoryError) as e:
            if self.verbose:
                print(f"  - 縮小なしMIPをスキップ: {type(e).__name__}: {e}")
            return np.nan

    def _build_cover_lists_from_packed_masks(self, rules: List[Dict], num_samples: int) -> List[np.ndarray]:
        """各サンプルをカバーするルール番号リストを、packed maskから復元する。"""
        num_rules = len(rules)
        nbytes_mask = len(rules[0]["mask_packed"])
        nbytes_key = (num_rules + 7) // 8
        keys = np.zeros((num_samples, nbytes_key), dtype=np.uint8)

        mask_packed_mat = np.empty((num_rules, nbytes_mask), dtype=np.uint8)
        for j, rule in enumerate(rules):
            mask_packed_mat[j, :] = np.frombuffer(rule["mask_packed"], dtype=np.uint8, count=nbytes_mask)

        for start in range(0, num_rules, MASK_BATCH_SIZE):
            end = min(start + MASK_BATCH_SIZE, num_rules)
            block_bits = np.unpackbits(mask_packed_mat[start:end], axis=1, bitorder="little")
            block_bits = block_bits[:, :num_samples].astype(bool, copy=False)

            for local_j, mask in enumerate(block_bits):
                if not mask.any():
                    continue
                j = start + local_j
                keys[mask, j >> 3] |= (1 << (j & 7))

        cover_lists: List[np.ndarray] = []
        zero_key = b"\x00" * nbytes_key
        for i in range(num_samples):
            key = keys[i].tobytes()
            if key == zero_key:
                cover_lists.append(np.empty(0, dtype=np.int32))
                continue
            bits = np.unpackbits(np.frombuffer(key, dtype=np.uint8), bitorder="little")[:num_rules]
            cover_lists.append(np.flatnonzero(bits).astype(np.int32, copy=False))

        return cover_lists

    def _build_variance_objective(self, model: gp.Model, z, rules: List[Dict]):
        sum_n_var = gp.quicksum(rules[j]["N_j_variance_j"] * z[j] for j in range(len(rules)))

        if isinstance(self.allocator, OptimalAllocator):
            sum_n_std = gp.quicksum(rules[j]["N_j_std_j"] * z[j] for j in range(len(rules)))
            aux_sum_std = model.addVar(lb=0.0, name="aux_sum_std")
            model.addConstr(aux_sum_std == sum_n_std, name="aux_constr_std")
            return (1.0 / self.sample_size) * (aux_sum_std * aux_sum_std) - sum_n_var

        return sum_n_var

    def _visualize_rules(self) -> "graphviz.Digraph":
        """選択されたOMTルールをGraphvizの有向グラフに変換する。"""
        check_is_fitted(self, "is_fitted_")

        dot = graphviz.Digraph(comment="OMT Rules")
        font = "Yu Gothic UI"
        dot.attr(dpi="300")
        dot.attr("node", fontname=font, fontsize="11", margin="0.12", shape="box", style="rounded,filled")
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

    def __str__(self) -> str:
        return f"OMT({self.binning_strategy})"
