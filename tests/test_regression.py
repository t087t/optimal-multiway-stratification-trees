"""Protect the numerical behavior and executable bodies of the original code."""

import ast
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from tests.capture import quiet_capture

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


class RegressionTests(unittest.TestCase):
    def test_original_function_bodies_are_unchanged(self):
        expected = json.loads((FIXTURES / "original_functions.json").read_text(encoding="utf-8"))
        actual = {}
        for path in (ROOT / "src").rglob("*.py"):
            for node in ast.parse(path.read_text(encoding="utf-8-sig")).body:
                if isinstance(node, ast.FunctionDef):
                    actual[f"{path.relative_to(ROOT).as_posix()}:{node.name}"] = node
                elif isinstance(node, ast.ClassDef):
                    for method in node.body:
                        if isinstance(method, ast.FunctionDef):
                            if path.stem.startswith("_omt_"):
                                key = f"src/estimator/omt.py:OMTEstimator.{method.name}"
                            else:
                                key = (
                                    f"{path.relative_to(ROOT).as_posix()}:{node.name}.{method.name}"
                                )
                            actual[key] = method
        for key, expected_hash in expected.items():
            with self.subTest(function=key):
                self.assertIn(key, actual)
                content = ast.dump(actual[key], include_attributes=False)
                self.assertEqual(hashlib.sha256(content.encode()).hexdigest(), expected_hash)

    def test_numerical_snapshot(self):
        expected = json.loads((FIXTURES / "numerical.json").read_text(encoding="utf-8"))
        actual = quiet_capture()

        def compare(left, right, path=""):
            if isinstance(left, dict):
                self.assertEqual(set(left), set(right), path)
                for key in left:
                    compare(left[key], right[key], f"{path}/{key}")
            elif isinstance(left, (list, tuple)):
                self.assertEqual(len(left), len(right), path)
                for i, (a, b) in enumerate(zip(left, right)):
                    compare(a, b, f"{path}/{i}")
            elif isinstance(left, float):
                np.testing.assert_allclose(left, right, rtol=1e-13, atol=1e-13, err_msg=path)
            else:
                self.assertEqual(left, right, path)

        compare(expected, actual)


if __name__ == "__main__":
    unittest.main()
