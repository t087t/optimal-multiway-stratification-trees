"""Command-line entry point for the paper experiments."""

import argparse
import hashlib
import importlib
import json
import platform
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from src.experiments.config import ExperimentConfig

ROOT = Path(__file__).resolve().parent
DATA_FILES = {"customer": "customer_data.csv", "seoulbike": "SeoulBikeData.csv"}


def parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the paper experiments.

    Returns:
        argparse.ArgumentParser: Parser with dataset, experiment, and output options.
    """
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", required=True, choices=DATA_FILES)
    result.add_argument(
        "--experiment",
        choices=["comparison", "coverage", "depth", "all"],
        default="comparison",
    )
    result.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data",
        help="Root containing customer/ and seoulbike/",
    )
    result.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    result.add_argument("--sample-size", type=int, default=100)
    result.add_argument("--n-trials", type=int, default=10_000)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--no-trees", action="store_true", help="Skip Graphviz image export")
    result.add_argument("--no-plots", action="store_true", help="Skip plots; retain CSV results")
    result.add_argument(
        "--check",
        action="store_true",
        help="Check input schema, dependencies and Gurobi without fitting",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """Prepare data, run the requested experiments, and record run metadata.

    Args:
        argv: Command-line arguments without the program name. None reads sys.argv.

    Returns:
        int: Zero for successful execution/checks; one for handled input, dependency, I/O, or
            runtime errors.

    Raises:
        SystemExit: If argument parsing handles --help or rejects an invalid command line.

    Notes:
        The supplementary sweeps keep their existing per-configuration exception handling.
        Unexpected errors from experiment code propagate after metadata is marked failed.
    """
    args = parser().parse_args(argv)
    try:
        config = ExperimentConfig(
            args.sample_size,
            args.n_trials,
            args.seed,
            not args.no_trees,
            not args.no_plots,
        )
        csv_path = args.data_dir / args.dataset / DATA_FILES[args.dataset]
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Dataset not found: {csv_path}. See README.md for the required files and columns."
            )

        import matplotlib

        matplotlib.use("Agg")
        import gurobipy as gp
        import numpy as np
        from sklearn.exceptions import ConvergenceWarning

        from src.experiments import datasets

        np.random.seed(config.random_state)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        data = getattr(datasets, f"prepare_{args.dataset}")(csv_path, config)
        if config.sample_size > min(len(data.y_train), len(data.y_test)):
            raise ValueError("sample-size exceeds the training or test population")
        if config.save_trees:
            import shutil

            if shutil.which("dot") is None:
                raise RuntimeError(
                    "Graphviz 'dot' was not found on PATH. Install Graphviz or use --no-trees."
                )
        try:
            with gp.Env(empty=True) as env:
                env.setParam("OutputFlag", 0)
                env.start()
        except gp.GurobiError as exc:
            raise RuntimeError(f"Gurobi license check failed: {exc}") from exc
        if args.check:
            print(
                f"Checks passed: {args.dataset}, train={len(data.y_train)}, test={len(data.y_test)}, Gurobi={gp.gurobi.version()}"
            )
            return 0

        from dataclasses import asdict

        out = args.output_dir / args.dataset / str(config.sample_size)
        out.mkdir(parents=True, exist_ok=True)
        versions = {}
        for package in (
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
            "gurobipy",
            "optbinning",
            "joblib",
            "matplotlib",
            "seaborn",
            "graphviz",
            "rich",
            "japanize-matplotlib",
        ):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = None
        metadata = {
            "dataset": args.dataset,
            "experiment": args.experiment,
            "config": asdict(config),
            "data_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": versions,
            "status": "running",
        }
        meta_path = out / f"run_{args.experiment}.json"

        def save_metadata():
            """Write the current run settings and status to the metadata JSON file.

            Returns:
                None: The enclosing metadata dictionary is written to meta_path.

            Raises:
                OSError: If the output files cannot be written.
            """
            meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        save_metadata()
        try:
            experiments = (
                ["comparison", "coverage", "depth"]
                if args.experiment == "all"
                else [args.experiment]
            )
            for experiment in experiments:
                importlib.import_module(f"src.experiments.{experiment}").run(data, config, out)
        except Exception as exc:
            metadata["status"] = "failed"
            save_metadata()
            if isinstance(exc, gp.GurobiError):
                raise RuntimeError(f"Gurobi solve failed: {exc}") from exc
            raise
        metadata["status"] = "completed"
        save_metadata()
        print(f"Results saved to {out}")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
