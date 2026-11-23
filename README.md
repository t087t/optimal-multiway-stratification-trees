# Optimal Multi-way Stratification Trees

Implementation of Optimal Multi-way Stratification Trees (OMST) for interpretable
stratified sampling and population-mean estimation. Includes quantile and optimal
binning, proportional and optimal allocation, and comparisons with Random, CUPED,
KMeans, KMeans-DT, sequential forward selection, and CART.

## Installation

Environment: Python 3.13.2, Gurobi 12.0.2, Windows 11.
A valid Gurobi license is required. For tree images, install Graphviz and add
`dot` to `PATH`, or use `--no-trees` to skip tree export.

```bash
git clone https://github.com/t087t/optimal-multiway-stratification-trees.git
cd optimal-multiway-stratification-trees
python -m pip install -r requirements.txt -c constraints.txt
```

## Data

Download the [Customer Purchases Behaviour dataset](https://www.kaggle.com/datasets/sanyamgoyal401/customer-purchases-behaviour-dataset)
and [Seoul Bike Sharing Demand dataset](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand),
and place the UTF-8 CSV files as follows:

```text
data/
├── customer/customer_data.csv
└── seoulbike/SeoulBikeData.csv
```

The loader expects these column names:

| Dataset | Target | Features |
| --- | --- | --- |
| Customer | `purchase_amount` | `age`, `income`, `gender`, `education`, `region`, `loyalty_status`, `purchase_frequency` |
| Seoul Bike | `Rented_Bike_Count` | `Hour`, `Temperature`, `Humidity`, `Wind_speed`, `Visibility`, `Dew_point_temperature`, `Solar_Radiation`, `Rainfall`, `Snowfall`, `Seasons`, `Holiday`, `Functioning_Day` |

Rename the Seoul download's headers to match the table before running.
Preprocessing is defined in [datasets.py](src/experiments/datasets.py).
Datasets are not included in this repository.

## Usage

Run the method comparison:

```bash
python main.py --dataset customer
python main.py --dataset seoulbike
```

Run coverage or depth experiments, or all three experiments:

```bash
python main.py --dataset customer --experiment coverage
python main.py --dataset customer --experiment depth
python main.py --dataset customer --experiment all
```

Default settings: sample size 100, 10,000 Monte Carlo trials, seed 42, and a 50/50
train/test split. The main comparison uses at most five strata and three features
per OMT rule. Each MIP has a 3,600-second time limit and uses one thread.

Use `--check` to check the input and environment, `--no-plots` to skip result plots,
and `--help` for all options. Experiment settings are in
[src/experiments/](src/experiments/).

## Outputs

Results are saved to `outputs/<dataset>/<sample-size>/`: CSV tables, tree and
result plots, and a JSON file recording the run settings and environment.
Use `--output-dir` to choose another location; repeating a run overwrites its files.
