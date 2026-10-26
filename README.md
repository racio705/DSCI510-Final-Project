# NYC Airbnb Long-Stay Quote Benchmark

A geospatial asking-price comparison tool with a **locked temporal holdout**, host-balanced gradient boosting, strong training-only comparable-price baselines, and calibrated uncertainty intervals. The original DSCI 510 coursework and exploratory analyses remain available as supporting material.

## Validated results

The evaluation protocol was [committed before downloading the holdout snapshots](docs/TEMPORAL_EVALUATION_PROTOCOL.md), at commit `b75e5a69daf625312ca2a4c582d3ddbf36657447`. October 2025 trains the point models, June 2026 calibrates intervals, and September 2026 is the final test. There is no test-based parameter search or threshold change.

### 1. The quote model beats two strong comparable-price baselines

The primary test contains **311 host accounts**, one price-independent hash-selected listing per account. Neither its listing ID nor host ID appears in **any raw October or June record**, including records missing prices. The June calibration sample contains 1,159 novel accounts.

| September strict unseen-account test | MAE / night | Relative MAE improvement |
| --- | ---: | ---: |
| Host-balanced gradient boosting, locked primary | **$96.03** | — |
| Training-only comparable listing medians | $114.73 | **16.3% lower** with model |
| Training-only comparable host medians | $111.09 | **13.6% lower** with model |

The corresponding family-adjusted paired-bootstrap intervals for improvement are **[7.80%, 24.87%]** and **[5.33%, 22.21%]**. Both comparables use a fixed neighborhood/product/capacity/bedroom/minimum-stay hierarchy with minimum support of 20 listings and five hosts per non-global cell.

Median absolute error is **$32.42**, median absolute percentage error **23.5%**, and **53.1%** of point estimates fall within ±25% of the observed ask. Nominal-80% intervals cover **257/311 = 82.6%** of observations; median interval width is **$168.65**. All six **researcher-chosen offline screening criteria** pass. This supports a comparison/screening tool with material individual uncertainty, not industry certification or evidence of automatic-pricing returns.

![Locked temporal holdout](figures/temporal_holdout.png)

### 2. The fixed accommodation-category price gap recurs later

For **entire homes, one bedroom, declared capacity two, minimum exactly 30 nights, subway straight-line distance ≤500 m**:

| Snapshot | Manhattan median / night | Brooklyn median / night | Brooklyn lower | Lower after each borough's largest account is excluded |
| --- | ---: | ---: | ---: | ---: |
| 2026-06-14 | $251.56 | $161.13 | **35.9%** | **22.3%** |
| 2026-09-14 | $240.57 | $163.76 | **31.9%** | **18.4%** |

Both periods meet the locked recurrence rule and sample floors. Accounts/listings can recur in the market tables; these are repeated temporal snapshots, not independent newly created properties. Quality, exact neighborhood and amenities remain unmatched, so the recurring benchmark is not a causal borough premium.

![Repeated market benchmark](figures/temporal_market_replication.png)

Read the [Chinese verified conclusions](docs/VERIFIED_CONCLUSIONS_ZH.md), [complete validation and subgroup failures](docs/TEMPORAL_VALIDATION.md), [executed validated notebook](notebooks/validated_benchmark.ipynb), or download/open the [portable notebook HTML](results/validated_benchmark.html). [Resume and interview wording](docs/PROJECT_GUIDE_ZH.md) separates the temporal results from the earlier internal CV.

## Use the local quote tool

After running temporal validation, supply JSON or CSV property features:

```bash
python src/quote_benchmark.py --input examples/quote_request.json --output results/quote_example.json
```

The example is an explicitly **hypothetical** Brooklyn one-bedroom, capacity-two property with minimum 30 nights. It produces **$144.18/night**, nominal interval **$83.23–$249.24**, and comparable medians **$153**, supported by 793 training listings from 697 accounts. These values demonstrate software behavior, not an observed listing or current booking offer.

The tool infers census context, borough and subway proximity from coordinates, handles missing optional fields with training-fitted imputation, reports comparable fallback/support, and retains the validation decision and interval limitations. It rejects target-price inputs and out-of-scope hotel or shorter-stay requests. Host IDs are never model inputs.

## Reproduce

Tested with **Python 3.12**. From the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python src/validate_temporal.py
python src/quote_benchmark.py --input examples/quote_request.json
python scripts/execute_validated_notebook.py
python -m unittest discover -s tests -v
```

The first validation run downloads/cache-verifies the exact Inside Airbnb, ACS and TIGER inputs, then fits the fixed point models, freezes both future prediction files, calibrates on June only, and scores September. Use `--offline` for exact cached reruns. The original MTA file is bundled. No Census API key is required. Historical archives can expire; restore the exact cache rather than silently substituting inputs. All source hashes and the protocol SHA are retained.

The portable validated notebook reads published aggregate artifacts and makes no network calls. Its source-level results are reproduced by the validation command. Serialized models are generated locally under ignored `data/models/`; listing-level data, host identities and predictions remain outside Git. Full dependency versions are recorded in `requirements-lock.txt`.

## Method and limits

The training scope is **17,100 positive-priced, non-hotel long-stay listings**, with no price-tail trimming. Features cover property capacity/type, minimum stay, review/rating, host portfolio count, coordinates and frozen Census context. Host-balanced fitting gives each training account equal total loss weight. Account IDs are used only for balancing, novelty and aggregation. Geometry uses EPSG:32618 and one-to-one NYC tract joins; source references remain ACS/TIGER 2023 and the frozen original MTA extract.

The primary claim concerns same-snapshot asking-price estimation using contemporaneous later attributes. It is not a forecast from October information alone, achieved booking prices, a property valuation, revenue or a causal transit intervention. The 311-account sample is only 11 above its predeclared sample floor. Queens private rooms exceed the 30% median-error target, and several other subgroups are too small; all are shown. Missing prices, approximate coordinates, unobserved quality and temporal drift restrict generalization. An equivalence claim for adding transit was **not** established under the locked ±2% margin.

## Supporting analyses and original coursework

The earlier October-only analysis includes spatial and host-group CV, grouped feature permutations, stay/product segments, an interactive map and account-composition sensitivities. Reproduce it with `python src/run_analysis.py`, then `python scripts/execute_notebook.py`; use `python src/derive_findings.py` to rerun only its supplementary findings. Read [market exploration](docs/MARKET_FINDINGS.md), [source provenance](docs/DATA_SOURCES.md) and [historical model results](docs/RESULTS.md).

Original sources are preserved under `archive/coursework/` and course PDFs under `doc/`. Their historical reported metrics are not substitutes for the locked temporal test. Code is MIT licensed; third-party data retains its publishers' terms.
