"""Reduced and unreduced Gurobi models for OMT rule selection."""

import time
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from src.allocator import OptimalAllocator, ProportionalAllocator

MASK_BATCH_SIZE = 4096
MIP_TIME_LIMIT_SECONDS = 3600
MIP_THREADS = 1


class OptimizationMixin:
    """Build and solve reduced and unreduced OMT selection models."""

    def _solve_mip(
        self, rules: List[Dict], num_samples: int, env: gp.Env
    ) -> Union[List[Dict], None]:
        """Solve the rule-selection model after aggregating equivalent row constraints.

        Args:
            rules: Candidate rule dictionaries with feature conditions, packed masks, and outcome
                statistics.
            num_samples: Number of training rows represented by each packed coverage mask.
            env: Started Gurobi environment used to create the model.

        Returns:
            list[dict] | None: Rules selected by an incumbent, or None if no solution is available.

        Raises:
            gurobipy.GurobiError: If model creation or optimization fails.
            MemoryError: If packed rule-membership arrays cannot be allocated.

        Notes:
            Updates solver diagnostics and actual training coverage. A time-limit incumbent is
            accepted even when optimality is unproven.
        """
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
        # Encode each row's candidate-rule membership compactly.
        nbytes_key = (num_rules + 7) // 8
        keys = np.zeros((num_samples, nbytes_key), dtype=np.uint8)

        mask_packed_mat = np.empty((num_rules, nbytes_mask), dtype=np.uint8)
        for j, r in enumerate(rules):
            mask_packed_mat[j, :] = np.frombuffer(
                r["mask_packed"], dtype=np.uint8, count=nbytes_mask
            )

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
                keys[mask, byte] |= 1 << bit

            del block_bits

        zero_key = b"\x00" * nbytes_key
        groups_bytes: Dict[bytes, List[int]] = defaultdict(list)
        for i in range(num_samples):
            b = keys[i].tobytes()
            if b == zero_key:
                continue
            groups_bytes[b].append(i)

        self.reduction_stats_["instances_after_const_reduction"] = len(groups_bytes)

        self.actual_covered_samples_ = 0
        self.actual_coverage_rate_ = 0.0

        if self.verbose:
            end_time_grouping = time.time()
            print(
                f"サンプルを {len(groups_bytes)} 個のグループに集約 (所要時間: {end_time_grouping - start_time_grouping:.2f}秒)"
            )
            print("\nMIPの最適化を開始")

        def bytes_to_rule_tuple(b: bytes) -> Tuple[int, ...]:
            """Decode the candidate-rule indices present in one packed group key.

            Args:
                b: Packed rule-membership bits for one group of training rows.

            Returns:
                tuple[int, ...]: Rule indices in increasing bit order.
            """
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

            sum_N_j_variance_j_z_j = gp.quicksum(
                rules[j]["N_j_variance_j"] * z[j] for j in range(len(rules))
            )
            variance_term = None

            if isinstance(self.allocator, OptimalAllocator):
                sum_N_j_std_j_z_j_expr = gp.quicksum(
                    rules[j]["N_j_std_j"] * z[j] for j in range(len(rules))
                )
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
            model.addConstr(
                covered_samples_expr >= self.min_coverage * num_samples, name="coverage"
            )

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

            self.reduction_stats_["mip_computation_time_sec"] = model.Runtime
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

            # Retain feasible incumbents when optimization ends before a proof of optimality.
            if model.SolCount > 0:
                selected_indices = [j for j, v in z.items() if v.X > 0.5]
                selected_set = set(selected_indices)

                actual_covered_samples = 0
                for rule_tuple, samples_in_group in groups.items():
                    if any(j in selected_set for j in rule_tuple):
                        actual_covered_samples += len(samples_in_group)

                actual_coverage_rate = (
                    (actual_covered_samples / num_samples) if num_samples > 0 else 0.0
                )
                self.actual_covered_samples_ = actual_covered_samples
                self.actual_coverage_rate_ = actual_coverage_rate

                if self.verbose:
                    status_text = "最適解"
                    if model.Status == GRB.TIME_LIMIT:
                        status_text = "準最適解 (時間上限)"
                    elif model.Status == GRB.SUBOPTIMAL:
                        status_text = "準最適解"

                    print(f"\nMIPの最適化が完了 ({status_text})")
                    print(
                        f"  - 選択されたルール数: {len(selected_indices)} (上限: {self.max_rules})"
                    )
                    print(
                        f"  - カバーされたサンプル数: {int(actual_covered_samples)} / {num_samples} ({actual_coverage_rate:.2%})"
                    )
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
        """Solve the unreduced model and measure Gurobi runtime.

        Args:
            rules: Candidate rule dictionaries with feature conditions, packed masks, and outcome
                statistics.
            num_samples: Number of training rows represented by each packed coverage mask.
            env: Started Gurobi environment used to create the model.

        Returns:
            float: Solver runtime in seconds, or NaN for no rules or a caught solver/memory failure.

        Notes:
            GurobiError and MemoryError are handled locally; a finite runtime does not imply that an
            incumbent exists.
        """
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
                        model.addConstr(
                            gp.quicksum(z[int(j)] for j in idx) <= 1,
                            name=f"packing_i[{i}]",
                        )

                model.addConstr(gp.quicksum(z) <= self.max_rules, name="max_rules")
                covered_samples_expr = gp.quicksum(
                    gp.quicksum(z[int(j)] for j in idx) for idx in cover_lists if idx.size
                )
                model.addConstr(
                    covered_samples_expr >= self.min_coverage * num_samples,
                    name="coverage",
                )

                model.optimize()

                if self.verbose:
                    status = "解あり" if model.SolCount > 0 else f"解なし(status={model.Status})"
                    print(f"  - 縮小なしMIP: {status}, Runtime={model.Runtime:.2f}秒")

                return float(model.Runtime)

        except (gp.GurobiError, MemoryError) as e:
            if self.verbose:
                print(f"  - 縮小なしMIPをスキップ: {type(e).__name__}: {e}")
            return np.nan

    def _build_cover_lists_from_packed_masks(
        self, rules: List[Dict], num_samples: int
    ) -> List[np.ndarray]:
        """Recover the rule indices covering each training row.

        Args:
            rules: Candidate rule dictionaries with feature conditions, packed masks, and outcome
                statistics.
            num_samples: Number of training rows represented by each packed coverage mask.

        Returns:
            list[ndarray]: One int32 rule-index array per training row; uncovered rows have an empty
                array.

        Raises:
            IndexError: If rules is empty.
            MemoryError: If the packed membership arrays cannot be allocated.
        """
        num_rules = len(rules)
        nbytes_mask = len(rules[0]["mask_packed"])
        # Encode each row's candidate-rule membership compactly.
        nbytes_key = (num_rules + 7) // 8
        keys = np.zeros((num_samples, nbytes_key), dtype=np.uint8)

        mask_packed_mat = np.empty((num_rules, nbytes_mask), dtype=np.uint8)
        for j, rule in enumerate(rules):
            mask_packed_mat[j, :] = np.frombuffer(
                rule["mask_packed"], dtype=np.uint8, count=nbytes_mask
            )

        for start in range(0, num_rules, MASK_BATCH_SIZE):
            end = min(start + MASK_BATCH_SIZE, num_rules)
            block_bits = np.unpackbits(mask_packed_mat[start:end], axis=1, bitorder="little")
            block_bits = block_bits[:, :num_samples].astype(bool, copy=False)

            for local_j, mask in enumerate(block_bits):
                if not mask.any():
                    continue
                j = start + local_j
                keys[mask, j >> 3] |= 1 << (j & 7)

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
        """Construct the allocator-dependent rule-selection objective.

        Args:
            model: Gurobi model to which auxiliary objective variables may be added.
            z (gurobipy.tupledict): Gurobi binary variables indexed by candidate rule number.
            rules: Candidate rule dictionaries with feature conditions, packed masks, and outcome
                statistics.

        Returns:
            gurobipy.LinExpr | gurobipy.QuadExpr: Variance objective without the common population
                normalization.

        Raises:
            gurobipy.GurobiError: If auxiliary variables or constraints cannot be added.
            ZeroDivisionError: If optimal allocation is used with sample_size equal to zero.

        Notes:
            Optimal allocation adds the sum-of-standard-deviations auxiliary variable; other
            allocators use the linear variance term.
        """
        sum_n_var = gp.quicksum(rules[j]["N_j_variance_j"] * z[j] for j in range(len(rules)))

        if isinstance(self.allocator, OptimalAllocator):
            sum_n_std = gp.quicksum(rules[j]["N_j_std_j"] * z[j] for j in range(len(rules)))
            aux_sum_std = model.addVar(lb=0.0, name="aux_sum_std")
            model.addConstr(aux_sum_std == sum_n_std, name="aux_constr_std")
            return (1.0 / self.sample_size) * (aux_sum_std * aux_sum_std) - sum_n_var

        return sum_n_var
