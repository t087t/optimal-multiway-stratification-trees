# Optimal Multi-way Stratification Trees
## Environment

The experiments reported in the paper were conducted using:

- Python 3.13.2
- Gurobi Optimizer 12.0.2
- Windows 11

A valid Gurobi license is required to solve the optimization problems.  
Graphviz must also be installed to generate decision-tree images.

## Installation

Clone this repository and install the required Python packages:

```bash
git clone https://github.com/t087t/optimal-multiway-stratification-trees.git
cd optimal-multiway-stratification-trees
pip install -r requirements.txt
```

## Main Experimental Settings

- Sample size: 100
- Monte Carlo trials: 10,000
- Random seed: 42
- Maximum number of strata in the main comparison: 5
- Maximum tree depth in the main comparison: 3
- Coverage lower bound: 1.0
- Quantile binning: 4 bins
- Gurobi time limit: 3,600 seconds
- Gurobi threads: 1

## Directory Structure

```text
.
├── README.md
├── requirements.txt
├── customer_data-rmse.ipynb
├── seoukbike.ipynb
└── src/
    ├── allocator.py
    └── estimator/
        ├── __init__.py
        ├── cart.py
        ├── coss.py
        ├── cuped.py
        ├── kmeansdt.py
        ├── lasso.py
        ├── mlrate.py
        ├── normal.py
        ├── omt.py            # Optimal Multi-way Stratification Tree estimator.
        ├── protocol.py
        ├── random.py
        └── sfs.py
```

## Experiment Details

### Seoul Bike Sharing Demand Dataset
https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand
- **Training Data**: 4,380 records (50% random split to avoid seasonal bias)
- **Test Data**: 4,380 records (50% random split)
- **Target Variable**: Rented Bike Count
- **Features**:
  - Continuous: Hour, Temperature (°C), Humidity (%), Wind speed (m/s), Visibility (10m), Dew point temperature (°C), Solar radiation (MJ/m2), Rainfall (mm), Snowfall (cm)
  - Categorical: Seasons, Holiday, Functioning Day

### Customer Purchases Behaviour Dataset
https://www.kaggle.com/datasets/sanyamgoyal401/customer-purchases-behaviour-dataset
- **Training Data**: 50,000 records (50% random split)
- **Test Data**: 50,000 records (50% random split)
- **Target Variable**: Purchase amount
- **Features**: 
  - Continuous: Age, Annual Income
  - Categorical: Gender, Education, Region, Loyalty Status, Purchase Frequency
 
Place the CSV files as follows:

```text
data/
├── customer/
│   └── customer_data.csv
└── seoulbike/
    └── SeoulBikeData.csv
```
