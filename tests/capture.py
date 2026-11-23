"""Deterministic numerical snapshot shared by the regression tests."""

import contextlib
import io

import numpy as np
import pandas as pd
from joblib import parallel_backend
from threadpoolctl import threadpool_limits

from src.allocator import (
    OptimalAllocator,
    ProportionalAllocator,
    compute_y_var,
    estimate_y_mean,
    estimate_y_rmse,
    estimate_y_var,
)
from src.estimator import (
    CARTEstimator,
    CUPEDEstimator,
    KMeansDTEstimator,
    NormalEstimator,
    OMTEstimator,
    RandomEstimator,
    SequentialForwardSelectionFEstimator,
    SequentialForwardSelectionTEstimator,
    SequentialForwardSelectionVarEstimator,
)


def capture():
    rng = np.random.RandomState(1234)
    X = rng.normal(size=(160, 3))
    y = 20 + 3 * X[:, 0] + X[:, 1] ** 2 + rng.normal(size=160)
    labels = np.repeat(np.arange(4), 40)
    result = {}
    for allocator in (ProportionalAllocator(), OptimalAllocator(1)):
        n = allocator.solve(y, labels, 4, 16)
        result[f"allocator/{allocator}"] = {
            "allocation": n.tolist(),
            "variance": float(compute_y_var(n, y, labels)),
            "mean": float(estimate_y_mean(n, y, labels, 42)),
            "mc_variance": float(estimate_y_var(n, y, labels, 32, 42)),
            "rmse": float(estimate_y_rmse(n, y, labels, y.mean(), 32, 42)),
        }
        common = dict(allocator=allocator, sample_size=16, n_trials=32, random_state=42)
        models = [
            CARTEstimator(max_leaf_nodes=4, **common),
            KMeansDTEstimator(n_clusters=4, max_leaf_nodes=4, **common),
        ]
        models += [
            NormalEstimator(n_clusters=4, clustering_method=method, **common)
            for method in ("KMeans", "GMM")
        ]
        for kind in (
            SequentialForwardSelectionVarEstimator,
            SequentialForwardSelectionFEstimator,
            SequentialForwardSelectionTEstimator,
        ):
            models.append(
                kind(
                    maximum_features_to_select=3,
                    n_clusters=4,
                    select_maximum_features=False,
                    **common,
                )
            )
        for i, model in enumerate(models):
            if isinstance(model, KMeansDTEstimator):
                model.fit(X, y, X_orig=X)
            else:
                model.fit(X, y)
            allocation = getattr(
                model,
                "sample_size_each_cluster",
                getattr(model, "sample_size_each_stratum_", None),
            )
            result[f"model/{allocator}/{i}/{model}"] = {
                "rmse": float(model.predict(X, y, true_mean=float(y.mean()))),
                "allocation": allocation.tolist(),
                "selected": getattr(model, "selected_features_index", None),
            }
    for model in (RandomEstimator(32, 16, 42), CUPEDEstimator(32, 16, 42)):
        model.fit(X, y)
        result[str(model)] = float(model.predict(X, y, true_mean=float(y.mean())))

    frame = pd.DataFrame({"value": X[:, 0], "group": np.where(X[:, 1] > 0, "B", "A")})
    for strategy in ("quantile", "optbinning"):
        model = OMTEstimator(
            OptimalAllocator(1),
            sample_size=16,
            num_bins=3,
            max_features_in_rule=2,
            min_samples_in_rule=8,
            verbose=False,
            binning_strategy=strategy,
        )
        frame_, target = model._validate_and_convert_input(frame, y)
        if strategy == "optbinning":
            model._fit_optbinning_process(frame_, target)
        rules = model._generate_rules(frame_, target)
        unique = model._reduce_decision_variables(rules)
        model.selected_rules_ = unique[:3]
        result[f"rules/{strategy}"] = {
            "rules": [{**r, "mask_packed": r["mask_packed"].hex()} for r in rules],
            "unique_count": len(unique),
            "labels": model._assign_labels(frame_).tolist(),
            "cover_lists": [
                v.tolist() for v in model._build_cover_lists_from_packed_masks(unique, len(y))
            ],
        }
    return result


def quiet_capture():
    # Cap test resources only. The implementation and per-trial seeds are unchanged.
    with (
        contextlib.redirect_stdout(io.StringIO()),
        threadpool_limits(limits=1),
        parallel_backend("threading", n_jobs=2),
    ):
        return capture()
