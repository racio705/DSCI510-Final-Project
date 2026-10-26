"""Build the portable review notebook for the locked temporal benchmark."""
from pathlib import Path
import nbformat as nbf


def create_notebook(root: Path) -> Path:
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = [
        md("""# NYC long-stay asking-price benchmark: locked temporal validation

The primary claims below use the protocol committed before downloading the June/September 2026 snapshots. October 2025 trains point models; June calibrates intervals; September tests unseen listing/host identities. This notebook uses published aggregate artifacts and embeds the final figures. It does not download data, fit models or choose thresholds."""),
        code("""from pathlib import Path
import json, re
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'results/temporal_validation.json').exists() and (p / 'src/nyc_airbnb').is_dir())
summary = json.loads((ROOT / 'results/temporal_validation.json').read_text(encoding='utf-8'))
metrics = pd.read_csv(ROOT / 'results/tables/temporal_holdout_metrics.csv')
primary_cohort = 'PRIMARY one listing per strict novel host'
primary_name = 'Host-balanced HGB without transit'
main = metrics[(metrics.cohort == primary_cohort) & (metrics.model == primary_name)].iloc[0]
display(Markdown(f"**Decision:** `{summary['decision']}`. "
                 f"Final holdout: **{int(main['n']):,} accounts**, one selected listing each. "
                 f"Calibration: **{summary['calibration_hosts']:,} accounts**."))
"""),
        md("## Conclusions and scope"),
        code("""report = (ROOT / 'docs/VERIFIED_CONCLUSIONS_ZH.md').read_text(encoding='utf-8')
# Images are embedded natively in the following cells, so the exported HTML remains portable.
report = re.sub(r'!\\[[^\\]]*\\]\\([^)]*\\)\\n?', '', report)
display(Markdown(report))
"""),
        md("## Locked temporal holdout"),
        code("""display(Image(filename=str(ROOT / 'figures/temporal_holdout.png')))
display(metrics[metrics.cohort == primary_cohort][
    ['model','n','mae_usd','median_ae_usd','median_ape_pct','coverage_80_pct','median_relative_width']].round(3))
"""),
        md("## Repeated category benchmark"),
        code("""display(Image(filename=str(ROOT / 'figures/temporal_market_replication.png')))
replication = pd.read_csv(ROOT / 'results/tables/temporal_market_replication.csv')
display(replication[replication.is_principal][
    ['period','n_manhattan','n_brooklyn','median_price_usd_manhattan',
     'median_price_usd_brooklyn','brooklyn_gap_pct','brooklyn_gap_excluding_largest_hosts_pct']].round(2))
"""),
        md("## Local quote-tool demonstration\n\nThe input is a hypothetical feature specification, not a listing or a booking offer. Run `python src/quote_benchmark.py --input examples/quote_request.json` after temporal validation to regenerate it."),
        code("""example = json.loads((ROOT / 'results/quote_example.json').read_text(encoding='utf-8'))
assert example['hypothetical'] is True
estimate = example['estimates'][primary_name]
display(pd.DataFrame([
    {'measure':'Primary point quote, USD/night','value':estimate['point_usd_per_night']},
    {'measure':'Nominal 80% lower bound','value':estimate['nominal_prediction_interval_80pct']['lower_usd']},
    {'measure':'Nominal 80% upper bound','value':estimate['nominal_prediction_interval_80pct']['upper_usd']},
    {'measure':'Comparable listing median','value':example['estimates']['comparable_listing_median']['point_usd_per_night']},
    {'measure':'Comparable host median','value':example['estimates']['comparable_host_median']['point_usd_per_night']},
]))
display(Markdown(example['interval_note'] + '\\n\\n' + example['scope_note']))
"""),
        md("## Integrity and predeclared decisions"),
        code("""assert summary['protocol_commit'] == 'b75e5a69daf625312ca2a4c582d3ddbf36657447'
assert int(main['n']) == summary['primary_test_hosts']
checks = pd.DataFrame([{'locked_criterion':key,'passed':value} for key,value in summary['gates'].items()])
display(checks)
comparisons = pd.read_csv(ROOT / 'results/tables/temporal_paired_intervals.csv')
for name in ['comparable_listing_median','comparable_host_median']:
    baseline = metrics[(metrics.cohort == primary_cohort) & (metrics.model == name)].iloc[0]
    expected = 100 * (1-main.mae_usd/baseline.mae_usd)
    observed = comparisons.loc[comparisons.contrast == name, 'estimate_pct'].iloc[0]
    assert abs(expected-observed) < 1e-8
display(comparisons.round(4))
manifest = json.loads((ROOT / 'results/temporal_source_manifest.json').read_text(encoding='utf-8'))
display(pd.DataFrame([{'snapshot':date,'bytes':record['bytes'],'sha256':record['sha256']}
                      for date,record in manifest['sources'].items()]))
print('Aggregate arithmetic checks passed. Source-level independence and errors were independently recomputed from frozen local predictions.')
"""),
        md("""## Reproduction

Use Python 3.12 and the pinned requirements. `python src/validate_temporal.py` downloads/verifies frozen sources, trains the fixed candidates, freezes predictions, calibrates and scores. `python scripts/execute_validated_notebook.py` rebuilds this executed review artifact and its HTML export. The application estimates contemporaneous asking prices from property/location inputs. It is not a forecast of achieved bookings, a causal intervention result or evidence of revenue gains.

The protocol, full model/baseline tables, subgroup failures, selection rates, prediction hashes and source hashes remain in the repository. Raw listing-level data, host identities and trained serialized objects are excluded from Git."""),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update({"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                              "language_info":{"name":"python"}})
    nbf.validate(notebook)
    destination=root/'notebooks/validated_benchmark.ipynb'
    destination.parent.mkdir(parents=True,exist_ok=True)
    nbf.write(notebook,destination)
    return destination
