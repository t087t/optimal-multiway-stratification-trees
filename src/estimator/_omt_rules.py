"""OMT binning, rule generation, and duplicate-coverage reduction."""

import itertools
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from optbinning import BinningProcess, ContinuousOptimalBinning


class RuleGenerationMixin:
    """Generate OMT conditions, candidate rules, and duplicate-mask reductions."""

    def _cumulative_binning(self, series: pd.Series, y: pd.Series) -> List[Tuple[float, float]]:
        """Enumerate threshold intervals for one numeric feature.

        Args:
            series: Values of one feature, in training-row order.
            y: Target values accepted by the helper; fitted optimal-bin splits are reused instead.

        Returns:
            list[tuple[float, float]]: Lower-open, upper-closed intervals excluding the full real
                line.

        Raises:
            ValueError: If the binning strategy or quantile-binning input is invalid.
        """
        quantiles = []

        if self.binning_strategy == "quantile":
            s = series.dropna().to_numpy()
            uniq = np.sort(np.unique(s))
            if len(uniq) <= self.num_bins:
                quantiles = uniq[1:-1]
            else:
                bins = pd.qcut(series, q=self.num_bins, retbins=True, duplicates="drop")[1]
                quantiles = bins[1:-1]
            if self.verbose:
                print(f"  - 特徴量 '{series.name}': 分位数ビニング (k={self.num_bins})")

        elif self.binning_strategy == "optbinning":
            splits = np.array(self.optb_splits_.get(series.name, []), dtype=float)
            splits = np.sort(np.unique(splits[~np.isnan(splits)]))
            quantiles = splits
            if self.verbose:
                print(
                    f"  - 特徴量 '{series.name}': 最適ビニング (max_n_bins={self.optb_max_n_bins})"
                )

        else:
            raise ValueError(f"未知のビニング戦略: {self.binning_strategy}")

        thresholds = np.concatenate(([-np.inf], np.sort(np.unique(quantiles)), [np.inf]))
        if self.verbose:
            threshold_str = np.array2string(
                np.round(thresholds, 2), formatter={"float_kind": lambda x: f"{x:.2f}"}
            )
            print(f"    - 閾値: {threshold_str}")

        if len(thresholds) < 2:
            return []

        # Exclude the universal interval so every numeric condition restricts coverage.
        intervals = [
            pair
            for pair in itertools.combinations(thresholds, 2)
            if not (pair[0] == -np.inf and pair[1] == np.inf)
        ]

        if self.verbose:
            print(f"    - 閾値の組み合わせから {len(intervals)} 個のルール候補を生成")

        return intervals

    def _generate_rules(self, X: pd.DataFrame, y: pd.Series) -> List[Dict[str, Any]]:
        """Enumerate supported feature conjunctions and their packed coverage masks.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Population outcomes of shape (n_samples,), aligned with the feature rows.

        Returns:
            list[dict]: Candidate rules in enumeration order, including population sizes and outcome
                statistics.

        Raises:
            RuntimeError: If optimal binning leaves no selected features.
            ValueError: If numeric feature values or the binning strategy are invalid.

        Notes:
            Updates feature_order_. Candidates below min_samples_in_rule are pruned.
        """
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
                raise RuntimeError(
                    "optbinningで選択された特徴量なし selection_criteriaの緩和が必要"
                )

            if self.verbose:
                print(f"  - optbinning: selected variables only -> {X.columns.tolist()}")

        n_samples = X.shape[0]
        y_values = y.to_numpy()
        nbytes = (n_samples + 7) // 8
        feature_types, X_converted = self._infer_feature_types(X)
        # Stable ordering also determines rule enumeration and equal-length tie handling.
        feature_names = sorted(
            X.columns.tolist(), key=lambda c: X_converted[c].nunique(dropna=True)
        )
        self.feature_order_ = feature_names

        atomic_conditions: List[List[Dict[str, Any]]] = []
        for col in feature_names:
            if self.binning_strategy == "optbinning" and feature_types[col] == "categorical":
                atomic_conditions.append(
                    self._build_optbinning_categorical_conditions(col, X_converted[col])
                )
                continue

            if feature_types[col] == "categorical":
                atomic_conditions.append(
                    self._build_singleton_categorical_conditions(col, X_converted[col])
                )
                continue

            atomic_conditions.append(self._build_numeric_conditions(col, X[col], y))

        rules: List[Dict[str, Any]] = []
        initial_mask = np.ones(n_samples, dtype=bool)

        def _build_rules_recursively(
            feat_idx_start: int,
            current_features: Dict[str, Any],
            current_mask: np.ndarray,
        ):
            """Append supported rule extensions in the established feature order.

            Args:
                feat_idx_start: First feature index eligible for extension of the current rule.
                current_features: Conditions already included in the current rule.
                current_mask: Training rows covered by the current conjunction.

            Returns:
                None: Candidate dictionaries are appended to the enclosing rules list.
            """
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
        """Pack a Boolean coverage mask using little-endian bit order.

        Args:
            mask_bool: Boolean coverage mask in training-row order.

        Returns:
            bytes: Packed mask padded to a complete final byte.
        """
        return np.packbits(mask_bool.astype(np.uint8, copy=False), bitorder="little").tobytes()

    def _infer_feature_types(self, X: pd.DataFrame) -> tuple[Dict[str, str], pd.DataFrame]:
        """Identify numeric/categorical columns and copy categorical values as strings.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            tuple: Feature-type mapping and a copied DataFrame with categorical string columns.
        """
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

    def _make_categorical_condition(
        self, feature: str, categories: set[str], values: np.ndarray
    ) -> Dict[str, Any]:
        """Build a category-membership condition with its row coverage mask.

        Args:
            feature: Name of the feature represented by the condition.
            categories: String categories included in the membership condition.
            values: String feature values in training-row order.

        Returns:
            dict: Feature name, sorted membership tuple, type, and Boolean mask.
        """
        ordered_categories = tuple(sorted(categories))
        return {
            "feature": feature,
            "condition": ("in", ordered_categories),
            "type": "categorical",
            "mask": np.isin(values, ordered_categories),
        }

    def _build_optbinning_categorical_conditions(
        self, feature: str, series: pd.Series
    ) -> List[Dict[str, Any]]:
        """Build membership conditions from fitted category groups and complements.

        Args:
            feature: Name of the feature represented by the condition.
            series: Values of one feature, in training-row order.

        Returns:
            list[dict]: Distinct nonempty, nonuniversal categorical conditions.
        """
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

    def _build_singleton_categorical_conditions(
        self, feature: str, series: pd.Series
    ) -> List[Dict[str, Any]]:
        """Build one membership condition for each observed category.

        Args:
            feature: Name of the feature represented by the condition.
            series: Values of one feature, in training-row order.

        Returns:
            list[dict]: Single-category conditions in observed category order.
        """
        values = series.to_numpy()
        categories = pd.Series(values).dropna().unique().tolist()
        return [
            self._make_categorical_condition(feature, {str(category)}, values)
            for category in categories
        ]

    def _build_numeric_conditions(
        self, feature: str, series: pd.Series, y: pd.Series
    ) -> List[Dict[str, Any]]:
        """Build numeric interval conditions and their training coverage masks.

        Args:
            feature: Name of the feature represented by the condition.
            series: Values of one feature, in training-row order.
            y: Population outcomes of shape (n_samples,), aligned with the feature rows.

        Returns:
            list[dict]: Interval conditions with lower-open, upper-closed masks.

        Raises:
            ValueError: If numeric values cannot be converted or binning fails.
        """
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

    def _fit_optbinning_process(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit optimal bins and retain the selected features and splits.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Population outcomes of shape (n_samples,), aligned with the feature rows.

        Returns:
            None: Fitted binning models, selected variables, and splits are stored on self.

        Raises:
            ValueError: If feature/target data or optimal-binning parameters are invalid.
        """
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
                monotonic_trend=(
                    self.optb_monotonic_trend if self.optb_monotonic_trend is not None else "auto"
                ),
            )
            optb.fit(X[col].to_numpy(), y.to_numpy())
            dict_optb[col] = optb

        self.binning_process_ = BinningProcess(
            variable_names=variable_names,
            verbose=False,
            selection_criteria={"quality_score": {"min": 0.01, "strategy": "highest", "top": 3}},
        )
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
        """Keep the shortest rule for each identical packed coverage mask.

        Args:
            rules: Candidate rule dictionaries with feature conditions, packed masks, and outcome
                statistics.

        Returns:
            list[dict]: Deduplicated rules in first-mask occurrence order.

        Notes:
            Equal-length ties retain the first rule, preserving the original enumeration order.
        """
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
            # Python min retains the first rule when lengths tie.
            best_rule = min(grouped_rules, key=lambda r: len(r["features"]))
            unique_rules.append(best_rule)

        if self.verbose:
            end_time = time.time()
            print(
                f"削減後のユニークなルール数: {len(unique_rules)}個 所要時間: {end_time - start_time:.2f}秒"
            )

        return unique_rules
