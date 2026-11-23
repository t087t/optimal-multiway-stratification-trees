"""Public OMT estimator, fit orchestration, labels, and RMSE evaluation."""

import gc
import time
from typing import List, Literal

import gurobipy as gp
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted

from src.allocator import (
    AllocatorProtocol,
    estimate_y_rmse,
)

from ._omt_rules import RuleGenerationMixin
from ._omt_solver import (
    OptimizationMixin,
)
from ._omt_visualization import RuleVisualizationMixin


class OMTEstimator(
    RuleGenerationMixin,
    OptimizationMixin,
    RuleVisualizationMixin,
    RegressorMixin,
    BaseEstimator,
):
    """Learn nonoverlapping stratification rules with mixed-integer optimization.

    Attributes:
        selected_rules_: Rule dictionaries selected during fit.
        n_rules_: Number of selected rules.
        labels_: Training-row rule labels; -1 denotes an uncovered row.
        sample_size_each_cluster: Integer allocation learned on covered training rows.
        reduction_stats_: Candidate counts, reduction counts, and solver diagnostics.
    """

    def __init__(
        self,
        allocator: "AllocatorProtocol",
        sample_size: int,
        binning_strategy: Literal["quantile", "optbinning"] = "quantile",
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
        """Store rule-learning, binning, allocation, and simulation settings.

        Args:
            allocator: Allocator used to determine integer sample counts for each stratum.
            sample_size: Total number of observations to sample without replacement.
            binning_strategy: Numeric binning strategy: "quantile" or "optbinning".
            categorical_variables: Names of categorical columns. None infers categories from
                nonnumeric dtypes.
            num_bins: Requested number of quantile bins for numeric features.
            max_features_in_rule: Maximum distinct features used by one candidate rule.
            max_rules: Maximum number of selected, nonoverlapping rules.
            min_coverage: Minimum fraction of training rows covered by the selected rules.
            min_samples_in_rule: Minimum training rows covered by each candidate rule.
            n_trials: Number of Monte Carlo sampling trials.
            verbose: Whether to print rule-generation and optimization diagnostics.
            random_state: Initial random seed for model fitting and Monte Carlo sampling.
            run_unoptimized_comparison: Whether to solve the unreduced model before the reduced
                model for timing comparison.
            optb_max_n_prebins: Maximum number of prebins passed to optimal binning.
            optb_min_prebin_size: Minimum population fraction in each prebin.
            optb_max_n_bins: Maximum final bins; None uses the optimal-binning default.
            optb_min_mean_diff: Minimum target-mean difference between adjacent bins.
            optb_max_pvalue: Maximum bin-comparison p-value; None disables this threshold.
            optb_gamma: Regularization strength passed to optimal binning.
            optb_monotonic_trend: Target-mean trend passed to optimal binning; None uses "auto".

        Returns:
            None: Settings are stored without generating or solving a model.

        Notes:
            Model and binning validation is deferred to fit. Constructor defaults are independent of
            the paper-comparison settings.
        """
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
        """Generate rules, solve the selection model, and learn an allocation.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Population outcomes of shape (n_samples,), aligned with the feature rows.

        Returns:
            OMTEstimator: This fitted estimator.

        Raises:
            ValueError: If training inputs or model parameters are invalid.
            RuntimeError: If optimal binning selects no usable features or no feasible rule
                combination is found.
            gurobipy.GurobiError: If the license, model construction, or optimization fails.
            AssertionError: If the configured allocator rejects its resulting sample counts.
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

            self.reduction_stats_["instances_before"] = n_samples
            self.reduction_stats_["paths_before"] = len(all_rules)

            if self.run_unoptimized_comparison:
                if self.verbose:
                    print(
                        f"\n比較のため、削減前の全ルール ({len(all_rules)}個) で縮小なしMIPを実行"
                    )
                self.reduction_stats_["mip_computation_time_unoptimized_sec"] = (
                    self._solve_mip_unoptimized(all_rules, n_samples, env)
                )

            if self.verbose:
                print("\n[ステップ2/4] 決定変数の削減")
            unique_rules = self._reduce_decision_variables(all_rules)
            self.reduction_stats_["paths_after_var_reduction"] = len(unique_rules)

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

            covered_mask = self.labels_ != -1
            if np.any(covered_mask):
                y_covered = y[covered_mask]
                labels_covered = self.labels_[covered_mask]

                start_time = time.time()
                self.sample_size_each_cluster = self.allocator.solve(
                    y_covered, labels_covered, self.n_rules_, self.sample_size
                )
                end_time = time.time()
                if self.verbose:
                    print(
                        f"  - サンプルサイズ割り当て完了 (所要時間: {end_time - start_time:.4f}秒)"
                    )
            else:
                if self.verbose:
                    print(
                        "  - 警告: カバーされたサンプルがないため、サンプルサイズ割り当てをスキップ"
                    )
                self.sample_size_each_cluster = np.zeros(self.n_rules_, dtype=int)

            self.is_fitted_ = True

            overall_end_time = time.time()
            if self.verbose:
                print(f"fitメソッド全体の所要時間: {overall_end_time - overall_start_time:.2f}秒")

            return self

    def _assign_labels(self, X: pd.DataFrame) -> NDArray[np.int_]:
        """Assign selected-rule indices to rows, leaving uncovered rows at -1.

        Args:
            X: DataFrame containing the feature names used by the selected rules.

        Returns:
            ndarray: Rule labels of shape (n_samples,).

        Raises:
            AttributeError: If selected rules are unavailable.
            KeyError: If a feature required by a selected rule is missing.
            ValueError: If a numeric rule feature cannot be converted to float.

        Notes:
            Rules are applied in stored order; a later matching rule overwrites an earlier label.
        """
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

    def predict(self, X: NDArray[np.float64], y: NDArray[np.float64], true_mean: float) -> float:
        """Estimate population-mean RMSE over repeated sampling trials.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Population outcomes of shape (n_samples,), aligned with the feature rows.
            true_mean: Mean of the full population against which estimates are evaluated.

        Returns:
            float: Monte Carlo RMSE, not a vector of row-level predictions.

        Raises:
            NotFittedError: If the estimator has not been fitted.
            ValueError: If inputs have invalid shapes or values, or an allocated draw exceeds its
                stratum population.

        Notes:
            Only covered rows are sampled. Returns NaN when no evaluation row is covered; true_mean
            still refers to the full population.
        """
        check_is_fitted(self, "is_fitted_")
        X_df, y_s = self._validate_and_convert_input(X, y)

        labels = self._assign_labels(X_df)

        # Keep uncovered rows out of sampling while retaining the full-population target.
        covered_mask = labels != -1

        if not np.any(covered_mask):
            print("警告: 予測対象データにカバーされたサンプルなし")
            return float("nan")

        y_covered = y_s.to_numpy()[covered_mask]
        labels_covered = labels[covered_mask]

        current_cluster_sizes = np.bincount(
            labels_covered, minlength=len(self.sample_size_each_cluster)
        )

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
        """Validate feature names and align outcomes with a copied DataFrame.

        Args:
            X (pandas.DataFrame | ndarray): Feature DataFrame or two-dimensional array using the
                fitted column order.
            y (ndarray): Population outcomes of shape (n_samples,), aligned with the feature rows.

        Returns:
            tuple: Feature DataFrame and target Series sharing the same row index.

        Raises:
            ValueError: If arrays are invalid, row counts differ, or column names/order differ from
                fit.
        """
        y_checked = check_array(y, ensure_2d=False, dtype="numeric")

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_checked = check_array(X, ensure_2d=True, dtype=None, force_all_finite=False)
            X_df = pd.DataFrame(X_checked, columns=self.feature_names_in_)

        if hasattr(self, "feature_names_in_") and self.feature_names_in_ != X_df.columns.tolist():
            raise ValueError("fit時とpredict/fit時で特徴量名が異なる")

        y_s = pd.Series(y_checked, name="target", index=X_df.index)
        return X_df, y_s

    def __str__(self) -> str:
        """Return the method label used in experiment tables.

        Returns:
            str: Human-readable method name.
        """
        return f"OMT({self.binning_strategy})"
