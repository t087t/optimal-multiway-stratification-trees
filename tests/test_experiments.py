"""Check the notebook migration, default settings, and command-line handling."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from main import main
from src.experiments.comparison import build_estimators
from src.experiments.config import ExperimentConfig
from src.experiments.datasets import prepare_customer, prepare_seoulbike
from tests.sample_data import customer_table, data_snapshot, seoulbike_table


class ExperimentTests(unittest.TestCase):
    def test_preprocessing_matches_original_notebooks(self):
        expected = json.loads(
            (Path(__file__).parent / "fixtures/preprocessing.json").read_text(encoding="utf-8")
        )
        for name, table, prepare in [
            ("customer", customer_table(), prepare_customer),
            ("seoulbike", seoulbike_table(), prepare_seoulbike),
        ]:
            with self.subTest(dataset=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "input.csv"
                table.to_csv(path, index=False)
                self.assertEqual(data_snapshot(prepare(path, ExperimentConfig())), expected[name])

    def test_comparison_order_and_defaults_match_notebook(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory) / "input.csv"
            customer_table().to_csv(path, index=False)
            data = prepare_customer(path, ExperimentConfig())
            models = build_estimators(data, ExperimentConfig())
        self.assertEqual(
            [str(model) for model in models],
            ["Random", "CUPED"]
            + ["OMT(quantile)", "OMT(optbinning)", "KMeans", "KMeans-DT", "SFS-KMeans-Var", "CART"]
            * 2,
        )
        self.assertEqual(
            [str(model.allocator) for model in models[2:]], ["Proportional"] * 6 + ["Optimal"] * 6
        )
        for model in models:
            self.assertEqual(
                (model.sample_size, model.n_trials, model.random_state), (100, 10000, 42)
            )
        for model in (models[2], models[3], models[8], models[9]):
            self.assertEqual(
                (
                    model.max_rules,
                    model.max_features_in_rule,
                    model.min_coverage,
                    model.min_samples_in_rule,
                ),
                (5, 3, 1.0, 6),
            )
            self.assertEqual(model.categorical_variables, data.features_list_cat)
        self.assertEqual(models[3].optb_max_n_bins, 20)
        self.assertEqual(models[3].optb_max_pvalue, 0.05)

    def test_missing_input_has_actionable_message(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.redirect_stderr(io.StringIO()) as error,
        ):
            code = main(["--dataset", "customer", "--data-dir", directory])
        self.assertEqual(code, 1)
        self.assertIn("Dataset not found", error.getvalue())

    def test_invalid_configuration(self):
        for kwargs in (
            {"sample_size": 0},
            {"n_trials": 0},
            {"random_state": -1},
            {"random_state": 2**32 - 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ExperimentConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
