# Optimal Multi-way Stratification Trees

Research code for learning interpretable strata and reducing the RMSE of a
population-mean estimator through stratified sampling. Optimal Multi-way
Stratification Trees (OMST, named `OMTEstimator` in the code) select nonoverlapping
rules using mixed-integer optimization.

The implementation includes quantile and optimal binning, proportional and optimal
allocation, and the comparison methods in the experiments: simple random sampling,
CUPED, KMeans, KMeans with a decision tree, sequential forward feature selection,
and CART.

## Installation

The paper's reported environment was **Python 3.13.2, Gurobi 12.0.2, and Windows 11**.
A valid Gurobi license is required. Install the Graphviz application and add `dot`
to `PATH` to export tree images; the Python `graphviz` package alone is insufficient.
You can disable tree export with `--no-trees`.

```bash
git clone https://github.com/t087t/optimal-multiway-stratification-trees.git
cd optimal-multiway-stratification-trees
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux (the paper was run on Windows)
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt -c constraints.txt
```

The dependency pins record the environment used for refactoring validation.
The original repository pinned only Gurobi; these files are **not a recovered
lockfile of the paper environment**. Run metadata records the actual installed
versions.

For license installation, follow the
[Gurobi instructions](https://support.gurobi.com/hc/en-us/articles/13207658935185-How-do-I-retrieve-an-Academic-Named-User-license).
Keep license files and keys outside Git. A size-limited license can run the small
solver tests but may not handle the paper experiments, particularly the quadratic
model for optimal allocation.

## Data

Download the datasets separately and place UTF-8 CSV files at these paths:

```text
data/
├── customer/customer_data.csv
└── seoulbike/SeoulBikeData.csv
```

Data and generated outputs are ignored by Git. The repository does not redistribute
the datasets.

| Dataset | Source | Target | Required feature columns |
| --- | --- | --- | --- |
| Customer Purchases Behaviour | [Kaggle](https://www.kaggle.com/datasets/sanyamgoyal401/customer-purchases-behaviour-dataset) | `purchase_amount` | `age`, `income`, `gender`, `education`, `region`, `loyalty_status`, `purchase_frequency` |
| Seoul Bike Sharing Demand | [UCI](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand) | `Rented_Bike_Count` | `Hour`, `Temperature`, `Humidity`, `Wind_speed`, `Visibility`, `Dew_point_temperature`, `Solar_Radiation`, `Rainfall`, `Snowfall`, `Seasons`, `Holiday`, `Functioning_Day` |

**The Seoul experiment expects the normalized headers above**, as did the original
notebook. The UCI download has different headers, including spaces and units, and
may use a different encoding. Prepare a UTF-8 copy with the required headers before
running. The loader does not guess mappings, change units, or reorder rows.
See [the input contract](docs/reproducibility.md#input-contract) for categorical
values and preprocessing details.

Both experiments use a seeded 50/50 random split. The reported dataset sizes were
100,000 customer rows and 8,760 Seoul rows before splitting; actual sizes depend
on the supplied files and missing values.

## Run

Check the input, dependencies, Graphviz, and Gurobi license without fitting:

```bash
python main.py --dataset customer --check
python main.py --dataset seoulbike --check
```

Run the main comparison:

```bash
python main.py --dataset customer
python main.py --dataset seoulbike
```

Run the supplementary experiments:

```bash
# Coverage lower bounds: 1.0, 0.95, 0.90, 0.85, 0.80
python main.py --dataset customer --experiment coverage

# OMT rule length / CART depth: 1, 2, 3, 4, 5
python main.py --dataset customer --experiment depth

# Comparison, coverage, then depth, in the original order
python main.py --dataset seoulbike --experiment all
```

`depth` also solves the unreduced MIP to measure the effect of problem reduction.
Each MIP retains the original 3,600-second limit; a full sweep can be expensive.

| Main-comparison setting | Value |
| --- | --- |
| Sample size | 100 |
| Monte Carlo trials | 10,000 |
| Random seed | 42 |
| Maximum strata / tree leaves | 5 |
| Maximum features per OMT rule | 3 |
| Coverage lower bound | 1.0 |
| Minimum training samples per rule / CART leaf | `int(0.05 * training_rows)` |
| Quantile bins | 4 |
| Gurobi time limit / threads | 3,600 seconds / 1 |

The main CART comparison limits **leaves**, not depth. Optimal-binning settings
and the different supplementary settings appear in
[reproducibility notes](docs/reproducibility.md#model-settings).

For a quicker diagnostic run on your own input:

```bash
python main.py --dataset customer --n-trials 100 --no-trees --no-plots
```

Changing `--n-trials`, `--sample-size`, or `--seed` changes the experiment and is
not a reproduction of default paper settings. `--no-trees` and `--no-plots` skip
image export while retaining computation and CSVs. `--data-dir` overrides the
root containing the `customer/` and `seoulbike/` subdirectories; `--output-dir`
overrides the output root. Run `python main.py --help` for all options.

## Outputs

Files are saved under `outputs/<dataset>/<sample-size>/`:

| File | Contents |
| --- | --- |
| `training_result_<sample-size>.csv` | Methods, allocations, selected features, scores, fitting time |
| `prediction_result_<sample-size>.csv` | Train/test RMSE and RMSE reduction rates |
| `coverage_results.csv` | Coverage sweep: selection bias, Monte Carlo variance, and MSE |
| `combined_experiment_results/rrr_comparison_results.csv` | Depth sweep: train/test RMSE reduction |
| `combined_experiment_results/computational_reduction_results.csv` | Rule/constraint reduction and MIP runtimes |
| `combined_experiment_results/mip_time_vs_depth_results.csv` | Reduced MIP time versus rule length |
| `run_<experiment>.json` | Options, input SHA-256, Python/packages, platform, execution status |

Trees and plots are saved alongside the tables or in experiment subdirectories.
Repeating an experiment overwrites its files; use a different `--output-dir` to
retain separate runs.

RMSE reduction rate is `100 * (1 - method_rmse / random_rmse)`.
Estimator `predict(X, y, true_mean)` returns a **Monte Carlo RMSE scalar**, rather
than a prediction vector. This research API is preserved.

## Structure

```text
main.py                         CLI and run metadata
src/
├── allocator.py                Allocation and Monte Carlo sampling
├── estimator/
│   ├── omt.py                  Public API and fitting orchestration
│   ├── _omt_rules.py           Binning, candidates, duplicate reduction
│   ├── _omt_solver.py          Reduced and unreduced Gurobi models
│   ├── _omt_visualization.py   Rule formatting and Graphviz export
│   └── ...                    Comparison estimators
└── experiments/
    ├── config.py               Runtime defaults
    ├── datasets.py             Preparation and aligned train/test views
    ├── comparison.py           Main comparison
    ├── coverage.py             Coverage sweep
    ├── depth.py                Depth and computational-reduction sweep
    └── plots/                 Figure export
docs/reproducibility.md          Preservation contract and validation scope
```

The notebooks were replaced by scripts, including their supplementary experiments
and plots, and remain available in Git history. Existing imports such as
`from src.estimator import OMTEstimator` remain valid.

See [implementation and reproducibility notes](docs/reproducibility.md) for the
preserved preprocessing, allocation, and model settings.
