"""Small actual Gurobi solves against the original implementation's outputs."""

import contextlib
import io
import json
import unittest
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from joblib import parallel_backend
from threadpoolctl import threadpool_limits

from src.allocator import OptimalAllocator, ProportionalAllocator
from src.estimator import OMTEstimator


def capture_solver():
    rng = np.random.RandomState(123)
    X = pd.DataFrame({"value": np.arange(64, dtype=float), "group": ["A"] * 32 + ["B"] * 32})
    y = 20 + np.arange(64) * 0.5 + rng.normal(size=64)
    result = {}
    for allocator in (ProportionalAllocator(), OptimalAllocator(1)):
        for strategy in ("quantile", "optbinning"):
            model = OMTEstimator(
                allocator,
                sample_size=8,
                num_bins=3,
                max_rules=3,
                max_features_in_rule=2,
                min_samples_in_rule=4,
                n_trials=32,
                random_state=42,
                verbose=False,
                binning_strategy=strategy,
                run_unoptimized_comparison=True,
            )
            model.fit(X, y)
            result[f"{allocator}/{strategy}"] = {
                "labels": model.labels_.tolist(),
                "allocation": model.sample_size_each_cluster.tolist(),
                "rules": [
                    {**r, "mask_packed": r["mask_packed"].hex()} for r in model.selected_rules_
                ],
                "rmse": float(model.predict(X, y, true_mean=float(y.mean()))),
                "objective": model.reduction_stats_["mip_obj_val"],
                "status": model.reduction_stats_["mip_status"],
                "unoptimized_solved": bool(
                    np.isfinite(model.reduction_stats_["mip_computation_time_unoptimized_sec"])
                ),
            }
            # Export is covered separately because Graphviz is optional for numerical tests.
    return result


def quiet_capture_solver():
    with (
        contextlib.redirect_stdout(io.StringIO()),
        threadpool_limits(limits=1),
        parallel_backend("threading", n_jobs=2),
    ):
        return capture_solver()


class SolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            with gp.Env(empty=True) as env:
                env.setParam("OutputFlag", 0)
                env.start()
        except gp.GurobiError as exc:
            raise unittest.SkipTest(f"Gurobi license unavailable: {exc}") from exc

    def test_small_solves_match_original(self):
        expected = json.loads(
            (Path(__file__).parent / "fixtures/solver.json").read_text(encoding="utf-8")
        )
        # JSON normalizes tuple-valued rule conditions in the same way on both sides.
        actual = json.loads(json.dumps(quiet_capture_solver()))
        self.assertEqual(expected.keys(), actual.keys())
        for key in expected:
            with self.subTest(model=key):
                for field in ("labels", "allocation", "rules", "status", "unoptimized_solved"):
                    self.assertEqual(expected[key][field], actual[key][field])
                for field in ("rmse", "objective"):
                    np.testing.assert_allclose(
                        expected[key][field], actual[key][field], rtol=1e-13, atol=1e-13
                    )

    def test_comparison_tables_match_original_notebook(self):
        from tests.pipeline import capture_comparison

        expected = json.loads(
            (Path(__file__).parent / "fixtures/comparison.json").read_text(encoding="utf-8")
        )
        self.assertEqual(expected, capture_comparison())


if __name__ == "__main__":
    unittest.main()
