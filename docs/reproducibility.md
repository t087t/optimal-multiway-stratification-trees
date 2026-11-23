# Reproducibility and preservation

## Scope

The baseline is commit `3e9f48e`. This refactoring makes experiments executable and
separates code responsibilities while preserving the research computation.
All retained estimator and allocator methods preserve their executable ASTs
(excluding docstrings), including defaults, annotations, arithmetic, iteration
order, and random-number generation.
Imports, comments, formatting, and private OMT method locations changed.

OMT uses three internal mixins. The public class remains
`src.estimator.omt.OMTEstimator`, with the same constructor and fitted attributes.
No optimizer, approximation, sampling algorithm, or tie-breaking rule is replaced.

The main order remains Random, CUPED, then quantile OMT, optimal-binning OMT,
KMeans, KMeans-DT, SFS-KMeans-Var, and CART for each of Proportional and Optimal
allocation. The unimplemented comparison references were removed as requested.

The unused estimator protocol and development-only tooling were removed.

Execution changes comprise the CLI, earlier schema/sample-size checks, saved
noninteractive plots, coverage CSV export, run metadata, and replacing notebooks
and Jupyter dependencies with ordinary scripts.

## Input contract

Preserve row order and units. CSVs use `pandas.read_csv` with default UTF-8 decoding
and the exact headers in the README. The loader does not download, sort, impute,
repair, or automatically normalize data.

| Customer column | Original categorical values |
| --- | --- |
| `gender` | `Male`, `Female` |
| `education` | `HighSchool`, `College`, `Bachelor`, `Masters` |
| `region` | `North`, `South`, `East`, `West` |
| `loyalty_status` | `Regular`, `Silver`, `Gold` |
| `purchase_frequency` | `frequent`, `occasional`, `rare` |

| Seoul column | Original categorical values |
| --- | --- |
| `Seasons` | `Spring`, `Summer`, `Autumn`, `Winter` |
| `Holiday` | `Holiday`, `No Holiday` |
| `Functioning_Day` | `Yes`, `No` |

Japanese feature/value translations are preserved because names and category
ordering can affect rule enumeration and solver ties. Encoded views are created
before splitting. Gender (customer) and Holiday/Functioning Day (Seoul) are binary;
the remaining categorical features are one-hot encoded in their original order.
Required target/encoded columns are selected and missing rows dropped before the
seeded 50/50 `train_test_split`.

**Encoded features retain `.astype(int)`, including continuous measurements.**
This truncation is deliberate preservation. CART and the KMeans-DT decision tree
receive the integer encoded view. KMeans, SFS, Random, and CUPED receive its
standardized view. `StandardScaler` is fitted only on training rows. OMT receives
unencoded, untruncated features aligned to the same train/test indices.

## Sampling and metrics

Sampling retains `np.random.RandomState`; trial `i` uses seed `random_state + i`.
Samples are drawn without replacement. Both train and test evaluation retain the
allocation learned on training data.

RMSE remains `sqrt(mean((estimated_mean - population_mean)**2))`.
The coverage sweep's MSE retains its distinct original definition: selection-bias
squared plus Monte Carlo variance with `ddof=1`. Selection bias is the covered
population mean minus the full population mean. It is not simply the squared
finite-trial RMSE. Uncovered rows are excluded from sampling, and existing
zero-allocation weight handling is unchanged.

Allocations exceeding test-stratum populations still raise an error. Sample sizes
are not clipped to force completion. Supplementary sweeps preserve their original
exception handling: coverage stores NaN rows for failed configurations; depth
omits failed configurations and prints errors. Inspect logs and row counts.

KMeans-DT retains its allocation attribute `sample_size_each_stratum_`. The
original comparison CSV looks up `sample_size_each_cluster`, so its informational
allocation field is empty for that method even though its RMSE uses the fitted
stratum allocation.

## Model settings

Main OMT uses at most five rules, three features per rule, full training coverage,
four quantile bins, and minimum rule size `int(0.05 * training_rows)`. Main CART
uses five leaves and that minimum leaf size; KMeans-DT also uses five leaves.
SFS stops when its score no longer improves (`select_maximum_features=False`),
with an upper feature-count limit of 100.

| Optimal-binning option | Value |
| --- | --- |
| `optb_max_n_prebins` | 20 |
| `optb_min_prebin_size` | 0.05 |
| `optb_max_n_bins` | 20 |
| `optb_min_mean_diff` | 0.0 |
| `optb_max_pvalue` | 0.05 |
| `optb_gamma` | 0.0 |
| `optb_monotonic_trend` | `auto` |

The binning process retains quality-score selection (minimum 0.01, top three
variables). The coverage sweep uses `[1.0, 0.95, 0.9, 0.85, 0.8]` with both
allocators and binning strategies. Supplementary experiments intentionally leave
`categorical_variables` unset, as in the notebooks, allowing type inference.

The depth sweep uses `[1, 2, 3, 4, 5]`. For OMT, it varies maximum features in a
rule; **the rule-count limit remains 32 at every depth**
(`2**max(MAX_DEPTH_LIST)`). CART varies `max_depth` without a five-leaf limit.
The unreduced MIP runs before the reduced one to measure runtimes.

Both MIP formulations retain a 3,600-second limit, one thread, and the estimator's
seed. Time-limited incumbents may be suboptimal; `reduction_stats_` retains solver
status, objective, bound, and gap. Hardware/environment/runtime changes can affect
the incumbent returned at the time limit. Exact equality of every large,
time-limited solve cannot be promised by a structural refactoring.

Graphviz IDs and colors retain the original Python `hash()` calls and can differ
between processes without changing strata or estimates. Timings and image bytes
are not numerical reproducibility checks.

## Refactoring verification

Before publication, all 82 original function/method definitions were checked
against commit `3e9f48e`. Local comparisons confirmed the original preprocessing,
sampling calculations, estimator results, and four small Gurobi solves. The full
comparison on an artificial customer CSV also matched the original notebook's
14 training rows and 28 prediction rows, excluding fitting time.

All three CLI workflows completed on artificial inputs for both datasets,
including CSV and figure export. These checks concern the refactoring; the
paper-scale experiments were not rerun. Verification scripts and artificial
fixtures are kept locally rather than included in this research-code repository.
