"""Capture full comparison tables on a small artificial population."""

import contextlib
import io
import json
import tempfile
from pathlib import Path

from joblib import parallel_backend
from threadpoolctl import threadpool_limits

from tests.sample_data import small_customer_table


def tables_snapshot(train, prediction):
    return {
        "training": json.loads(
            train.drop(columns="elapsed_time").to_json(
                orient="records", double_precision=12, force_ascii=False
            )
        ),
        "prediction": json.loads(
            prediction.to_json(orient="records", double_precision=12, force_ascii=False)
        ),
    }


def capture_comparison():
    from src.experiments.comparison import run
    from src.experiments.config import ExperimentConfig
    from src.experiments.datasets import prepare_customer

    with (
        tempfile.TemporaryDirectory() as directory,
        contextlib.redirect_stdout(io.StringIO()),
        threadpool_limits(limits=1),
        parallel_backend("threading", n_jobs=2),
    ):
        path = Path(directory)
        csv = path / "input.csv"
        small_customer_table().to_csv(csv, index=False)
        config = ExperimentConfig(sample_size=16, n_trials=8, save_trees=False, save_plots=False)
        data = prepare_customer(csv, config)
        train, prediction = run(data, config, path / "output")
        return tables_snapshot(train, prediction)
