"""Render the protocol-locked temporal results without refitting or selecting.

This reader is invoked only after temporal predictions and evaluation outputs
have been frozen. It publishes every adoption gate, secondary candidates,
market support checks and subgroup limitations, including unfavorable results.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd

PROTOCOL_COMMIT = "b75e5a69daf625312ca2a4c582d3ddbf36657447"
PRIMARY_COHORT = "PRIMARY one listing per strict novel host"
PRIMARY_MODEL = "Host-balanced HGB without transit"
MODELS = [PRIMARY_MODEL, "Host-balanced HGB with transit",
          "Listing-weighted HGB without transit",
          "comparable_listing_median", "comparable_host_median"]
MODEL_LABELS = {
    PRIMARY_MODEL: "Primary: host-balanced HGB\nwithout transit",
    MODELS[1]: "Secondary: host-balanced HGB\nwith transit",
    MODELS[2]: "Secondary: listing-weighted HGB\nwithout transit",
    MODELS[3]: "Baseline: comparable\nlisting medians",
    MODELS[4]: "Baseline: comparable\nhost medians",
}
BLUE, ORANGE, INK = "#23657d", "#c87736", "#26333b"


def _number(value, digits=2, prefix="", suffix="") -> str:
    if value is None or pd.isna(value):
        return "unavailable"
    return f"{prefix}{float(value):,.{digits}f}{suffix}"


def _money(value) -> str:
    return _number(value, prefix="$")


def _percent(value, digits=1) -> str:
    return _number(value, digits, suffix="%")


def _table(headers: list[str], rows: list[list]) -> str:
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", "<br>")
    lines = ["| " + " | ".join(map(cell, headers)) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    return "\n".join(lines + ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def _bool_column(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1"])


def _style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "text.color": INK, "axes.labelcolor": INK,
                         "axes.titlecolor": INK, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#98a6ae",
                         "xtick.color": INK, "ytick.color": INK,
                         "axes.axisbelow": True, "figure.facecolor": "white"})


def _holdout_figure(primary: pd.DataFrame, summary: dict, destination: Path) -> None:
    data = primary.set_index("model").loc[MODELS]
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True,
                             gridspec_kw={"width_ratios": [1.15, 1, 1]})
    fig.subplots_adjust(left=0.25, right=0.98, top=0.76, bottom=0.21, wspace=0.18)
    y = np.arange(len(data))
    colors = [BLUE, "#91a5ae", "#91a5ae", "#c2ccd1", "#c2ccd1"]
    definitions = [
        ("mae_usd", "Mean absolute error", "USD/night; lower is better"),
        ("median_ape_pct", "Median absolute percentage error", "Target ≤30%; lower is better"),
        ("coverage_80_pct", "Observed 80% interval coverage", "Target ≥75%; nominal level 80%"),
    ]
    for ax, (column, title, xlabel) in zip(axes, definitions):
        values = data[column].to_numpy(dtype=float)
        ax.barh(y, values, height=0.56, color=colors)
        finite = values[np.isfinite(values)]
        upper = max(float(finite.max()) * 1.25, 1) if len(finite) else 1
        if column == "median_ape_pct":
            upper = max(upper, 36)
            ax.axvline(30, color=INK, linestyle="--", linewidth=1)
        if column == "coverage_80_pct":
            upper = 105
            ax.axvline(75, color=INK, linestyle="--", linewidth=1)
            ax.axvline(80, color="#75858e", linestyle=":", linewidth=1)
        ax.set_xlim(0, upper)
        for position, value in zip(y, values):
            if np.isfinite(value):
                label = _money(value) if column == "mae_usd" else _percent(value)
                ax.text(value + upper * 0.015, position, label, va="center", fontsize=9,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7})
            else:
                ax.text(0.02, position, "unavailable", transform=ax.get_yaxis_transform(), va="center")
        ax.set_title(title, loc="left", fontsize=11, pad=14)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.grid(axis="x", color="#e3e8eb")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}") if column == "mae_usd"
                                          else PercentFormatter(xmax=100, decimals=0))
    axes[0].set_yticks(y, [MODEL_LABELS[name] for name in MODELS])
    axes[0].invert_yaxis()
    fig.suptitle("Temporal holdout performance on unseen host accounts", x=0.02,
                 y=0.96, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.02, 0.885,
             f"September 2026: {int(summary['primary_test_hosts']):,} hosts, one listing per host. "
             f"June calibration: {int(summary['calibration_hosts']):,} hosts.", fontsize=11)
    fig.text(0.02, 0.055,
             "The primary hosts and listings are absent from all raw October and June records. "
             "All five candidates are shown; secondary models do not replace the locked primary.\n"
             "October-only fitting; June-only interval calibration. Dashed targets are researcher-chosen screening criteria. "
             "Temporal drift removes an exchangeability guarantee.", fontsize=9, color="#52626c", linespacing=1.6)
    fig.savefig(destination, dpi=180, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def _market_figure(market: pd.DataFrame, destination: Path) -> None:
    periods = ["2025-10-01", "2026-06-14", "2026-09-14"]
    selected = market[_bool_column(market["is_principal"])].set_index("period")
    price_columns = ["median_price_usd_manhattan", "median_price_usd_brooklyn"]
    prices = selected[price_columns].to_numpy(dtype=float)
    largest_price = float(np.nanmax(prices)) if np.isfinite(prices).any() else 1
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.8), sharex=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.72, bottom=0.32, wspace=0.24)
    for ax, period in zip(axes, periods):
        if period not in selected.index:
            ax.text(0.1, 0.5, "Required period unavailable", transform=ax.transAxes)
            ax.set_title(period)
            continue
        row = selected.loc[period]
        values = [row[column] for column in price_columns]
        ax.barh([0, 1], values, height=0.48, color=[BLUE, ORANGE])
        for y, borough, value in zip([0, 1], ["manhattan", "brooklyn"], values):
            label = f"{_money(value)}\nn={int(row[f'n_{borough}']):,}; hosts={int(row[f'hosts_{borough}']):,}"
            ax.text(value + largest_price * 0.025, y, label, va="center", fontsize=9)
        ax.set_yticks([0, 1], ["Manhattan", "Brooklyn"])
        ax.invert_yaxis()
        ax.set_xlim(0, largest_price * 1.8)
        ax.set_ylim(1.6, -0.6)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.grid(axis="x", color="#e3e8eb")
        role = "Historical context" if period == periods[0] else "Future replication"
        ax.set_title(f"{period}\n{role}", loc="left", fontsize=11, pad=15)
        supported = str(row["sample_gate_met"]).lower() in ["true", "1"]
        note = (f"BK-minus-Manhattan gap: {_percent(row['brooklyn_gap_pct'])}\n"
                f"After largest-host exclusion: {_percent(row['brooklyn_gap_excluding_largest_hosts_pct'])}\n"
                f"≥100 listings and ≥30 hosts each: {'met' if supported else 'NOT MET'}")
        ax.text(0, -0.29, note, transform=ax.transAxes, va="top", fontsize=10,
                linespacing=1.55, color=INK if supported else ORANGE)
        ax.set_xlabel("Median advertised nightly price (USD)")
    fig.suptitle("Fixed-product borough benchmark across snapshots", x=0.02, y=0.97,
                 ha="left", fontsize=17, fontweight="bold")
    fig.text(0.02, 0.88,
             "Entire home · one bedroom · capacity exactly two · minimum exactly 30 nights · subway ≤500 m", fontsize=11)
    fig.text(0.02, 0.035,
             "Locked recurrence criterion: Brooklyn ≥10% lower in both 2026 periods, with a lower median after largest-host exclusion.\n"
             "Common category filters do not match quality or neighborhood. Listing overlap across snapshots is possible. "
             "Sources: Inside Airbnb and frozen MTA/Census references.", fontsize=9, color="#52626c", linespacing=1.6)
    fig.savefig(destination, dpi=180, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)


def _gate_rows(summary: dict, main: pd.Series, contrasts: pd.DataFrame, chinese=False) -> list[list]:
    contrast = contrasts.set_index("contrast")
    def statistic(key, column):
        return contrast.loc[key, column] if key in contrast.index else np.nan
    lower = lambda key: statistic(key, "family_ci_lower_pct")
    upper = lambda key: statistic(key, "family_ci_upper_pct")
    estimate = lambda key: statistic(key, "estimate_pct")
    definitions = [
        ("calibration_and_test_at_least_300_hosts", "校准及测试独立账号数" if chinese else "Calibration and test hosts",
         f"{summary['calibration_hosts']:,} / {summary['primary_test_hosts']:,}", "各 ≥300" if chinese else "Each ≥300"),
        ("beats_listing_comparables_by_10pct_with_positive_interval", "优于挂牌中位数基线" if chinese else "Improvement over listing comparables",
         f"{_percent(estimate('comparable_listing_median'))}; CI [{_number(lower('comparable_listing_median'))}%, {_number(upper('comparable_listing_median'))}%]",
         "降幅 ≥10%，区间下界 >0" if chinese else "Point improvement ≥10%; CI lower >0"),
        ("beats_host_comparables_by_10pct_with_positive_interval", "优于房东中位数基线" if chinese else "Improvement over host comparables",
         f"{_percent(estimate('comparable_host_median'))}; CI [{_number(lower('comparable_host_median'))}%, {_number(upper('comparable_host_median'))}%]",
         "降幅 ≥10%，区间下界 >0" if chinese else "Point improvement ≥10%; CI lower >0"),
        ("median_ape_at_most_30pct", "绝对百分比误差中位数" if chinese else "Median absolute percentage error",
         _percent(main['median_ape_pct']), "≤30%"),
        ("coverage_80_at_least_75pct", "名义 80% 区间实际覆盖率" if chinese else "Observed nominal-80% interval coverage",
         _percent(main['coverage_80_pct']), "≥75%"),
        ("median_relative_interval_width_at_most_1_5", "相对区间宽度中位数" if chinese else "Median relative interval width",
         _number(main['median_relative_width'], 3), "≤1.5"),
    ]
    rows = []
    for key, label, actual, criterion in definitions:
        if key not in summary['gates']:
            raise ValueError(f"Required protocol gate is absent: {key}")
        passed = summary['gates'][key]
        rows.append([label, actual, criterion,
                     ("通过" if passed else "未通过") if chinese else ("PASS" if passed else "FAIL")])
    return rows


def _subgroup_rows(subgroups: pd.DataFrame, chinese=False) -> list[list]:
    rows = []
    for _, primary in subgroups[subgroups['model'].eq(PRIMARY_MODEL)].iterrows():
        same = subgroups[subgroups['borough'].eq(primary['borough']) & subgroups['room_type'].eq(primary['room_type'])]
        base = same[same['model'].isin(MODELS[-2:])]['mae_usd']
        flags = []
        if primary['n'] < 30:
            flags.append("样本 <30" if chinese else "n<30")
        if len(base) and (base <= primary['mae_usd']).any():
            flags.append("未优于至少一个基线" if chinese else "MAE not better than ≥1 baseline")
        if primary['median_ape_pct'] > 30:
            flags.append("误差中位数 >30%" if chinese else "median APE>30%")
        if primary['coverage_80_pct'] < 75:
            flags.append("覆盖率 <75%" if chinese else "coverage<75%")
        rows.append([primary['borough'], primary['room_type'], int(primary['n']),
                     _money(primary['mae_usd']), _percent(primary['median_ape_pct']),
                     _percent(primary['coverage_80_pct']), "; ".join(flags) or ("未触发所列诊断标记" if chinese else "No listed diagnostic flag")])
    return rows


def render_temporal_report(root: Path | str) -> dict[str, str]:
    """Create the final English/Chinese readout and two figures from frozen outputs."""
    root = Path(root)
    summary = json.loads((root / 'results/temporal_validation.json').read_text(encoding='utf-8'))
    if summary['protocol_commit'] != PROTOCOL_COMMIT:
        raise ValueError("Temporal results refer to a different protocol commit")
    if summary['decision'] == 'met_predeclared_offline_criteria' and not all(summary['gates'].values()):
        raise ValueError("Adoption decision conflicts with reported gate values")
    metrics = pd.read_csv(root / 'results/tables/temporal_holdout_metrics.csv')
    primary = metrics[metrics['cohort'].eq(PRIMARY_COHORT)].copy()
    if primary['model'].duplicated().any() or not set(MODELS).issubset(primary['model']):
        raise ValueError("All locked primary and secondary candidates must be reported once")
    if not primary['n'].eq(primary['n_hosts']).all():
        raise ValueError("Primary temporal cohort must contain one listing per host")
    main = primary[primary['model'].eq(PRIMARY_MODEL)].iloc[0]
    if int(main['n_hosts']) != int(summary['primary_test_hosts']):
        raise ValueError("Summary and primary metrics have different host counts")
    contrasts = pd.read_csv(root / 'results/tables/temporal_paired_intervals.csv')
    market = pd.read_csv(root / 'results/tables/temporal_market_replication.csv')
    replication = json.loads((root / 'results/temporal_market_replication.json').read_text(encoding='utf-8'))
    market_main = market[_bool_column(market['is_principal'])].copy()
    subgroups = pd.read_csv(root / 'results/tables/temporal_holdout_subgroups.csv')
    docs, figures = root / 'docs', root / 'figures'
    docs.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    _style()
    _holdout_figure(primary, summary, figures / 'temporal_holdout.png')
    _market_figure(market, figures / 'temporal_market_replication.png')

    english_decisions = {
        'met_predeclared_offline_criteria': 'The locked primary model met every predeclared offline screening criterion.',
        'criteria_not_met': 'The locked primary model did not meet every predeclared offline screening criterion.',
        'insufficient_independent_sample': 'The independent-sample gate was not met; the protocol does not support an adoption decision.',
    }
    chinese_decisions = {
        'met_predeclared_offline_criteria': '主模型通过了全部预先锁定的离线筛选标准。',
        'criteria_not_met': '主模型未通过全部预先锁定的离线筛选标准。',
        'insufficient_independent_sample': '独立样本量未达到锁定门槛，本次结果不足以作出采用决策。',
    }
    metric_rows = [[MODEL_LABELS[row.model].replace('\n', ' '), _money(row.mae_usd),
                    _money(row.median_ae_usd), _percent(row.median_ape_pct), _percent(row.within_25pct),
                    _percent(row.coverage_80_pct), _number(row.median_relative_width, 3)]
                   for row in primary.set_index('model').loc[MODELS].reset_index().itertuples()]
    contrast_rows = []
    for row in contrasts.itertuples():
        label = 'Primary MAE improvement vs ' + row.contrast if row.contrast != 'transit_relative_mae' else 'Transit vs no-transit relative MAE change'
        contrast_rows.append([label, _percent(row.estimate_pct, 2),
                              f"[{_number(row.family_ci_lower_pct)}%, {_number(row.family_ci_upper_pct)}%]"])
    market_rows, concentration_rows = [], []
    for row in market_main.itertuples():
        market_rows.append([row.period, f"{_money(row.median_price_usd_manhattan)} (n={row.n_manhattan:,}; hosts={row.hosts_manhattan:,})",
                            f"{_money(row.median_price_usd_brooklyn)} (n={row.n_brooklyn:,}; hosts={row.hosts_brooklyn:,})",
                            _percent(row.brooklyn_gap_pct, 2), _percent(row.brooklyn_gap_excluding_largest_hosts_pct, 2),
                            row.period_status.replace('_', ' ')])
        for borough in ['manhattan', 'brooklyn']:
            concentration_rows.append([row.period, borough.title(),
                                       int(getattr(row, f'largest_host_n_{borough}')),
                                       _percent(getattr(row, f'largest_host_share_pct_{borough}')),
                                       _money(getattr(row, f'median_excluding_largest_host_usd_{borough}')),
                                       _money(getattr(row, f'median_of_host_medians_usd_{borough}'))])
    quality_rows = []
    for period, quality in summary['snapshot_quality'].items():
        raw, missing = quality['raw_listings'], quality['missing_or_invalid_price']
        quality_rows.append([period, f"{raw:,}", f"{missing:,}", _percent(100 * missing / raw),
                             f"{quality['eligible_long_stay_listings']:,}",
                             f"{quality.get('listings_missing_income', 0):,}"])
    cohort_rows = []
    for row in metrics[metrics['model'].eq(PRIMARY_MODEL)].itertuples():
        cohort_rows.append([row.cohort, int(row.n), int(row.n_hosts), _money(row.mae_usd),
                            _percent(row.median_ape_pct), _percent(row.coverage_80_pct)])
    equivalence_rows = contrasts[contrasts['contrast'].eq('transit_relative_mae')]
    equivalence = equivalence_rows.iloc[0] if len(equivalence_rows) else pd.Series({
        'estimate_pct': np.nan, 'family_ci_lower_pct': np.nan, 'family_ci_upper_pct': np.nan})
    if not summary['gates']['calibration_and_test_at_least_300_hosts']:
        equiv_text = 'The independent-sample gate was not met, so the protocol does not support an equivalence claim.'
        equiv_zh = '独立样本门槛未达到，因此按方案不能作出等效结论。'
    else:
        equiv_text = ('The full contrast interval lies within the predeclared ±2% margin; this holdout supports practical predictive equivalence under that definition.'
                  if summary['transit_practically_equivalent_within_2pct'] else
                  'Practical predictive equivalence was not established: the interval does not fit inside the predeclared ±2% margin. A lack of superiority is not evidence of equivalence.')
        equiv_zh = ('区间完全位于预先设定的 ±2% MAE 范围内，本次留出评估支持该定义下的实用预测等效。'
                if summary['transit_practically_equivalent_within_2pct'] else
                '区间未完全落入预设 ±2% 范围，未能证明实用预测等效。不能把“没有显示优势”写成“已证实没有作用”。')
    replication_status = replication['decision']['status']
    recurrence_en = {
        'replicated': 'The fixed descriptive benchmark met its recurrence criteria in both future snapshots.',
        'not_replicated': 'The fixed descriptive benchmark did not meet its recurrence criteria in both future snapshots.',
        'insufficient_sample': 'The fixed descriptive benchmark has insufficient support in at least one required future snapshot.',
    }[replication_status]
    recurrence_zh = {
        'replicated': '固定住宿类别的区域报价差异在两个未来快照中均通过了预先设定的重复检验标准。',
        'not_replicated': '固定住宿类别的区域报价差异未在两个未来快照中同时通过重复检验标准。',
        'insufficient_sample': '至少一个未来快照的比较样本不足，不能确认固定报价基准重复成立。',
    }[replication_status]
    margin = int(summary['primary_test_hosts']) - 300
    sample_margin_en = (f'The final sample is only {margin:,} accounts above the declared 300-account floor'
                        if margin >= 0 else f'The final sample is {-margin:,} accounts below the declared 300-account floor')
    sample_margin_zh = (f'测试集只比事先设定的 300 个账号门槛多 {margin:,} 个'
                        if margin >= 0 else f'测试集比事先设定的 300 个账号门槛少 {-margin:,} 个')
    market_rows_zh = [row[:-1] + [{
        'historical context': '历史背景，不计入重复检验',
        'meets period criteria': '通过该期标准',
        'does not meet period criteria': '未通过该期标准',
        'insufficient sample': '样本不足',
    }.get(row[-1], row[-1])] for row in market_rows]

    english = f"""# Frozen temporal validation

{english_decisions[summary['decision']]} The September primary sample contains **{summary['primary_test_hosts']:,} host accounts**, one listing per host, unseen in all raw October and June records. The June-only calibration sample contains {summary['calibration_hosts']:,} accounts. {sample_margin_en}; conclusions remain conditional on this historical sample.

The [protocol](TEMPORAL_EVALUATION_PROTOCOL.md) was committed before the future downloads as `{PROTOCOL_COMMIT}`. This is a pre-specified retrospective holdout evaluation, not a registered experiment or a causal study.

## Decision and every locked criterion

{_table(['Criterion', 'Observed', 'Locked requirement', 'Result'], _gate_rows(summary, main, contrasts))}

These are researcher-chosen engineering targets for offline screening. They are not industry certification or demonstrated commercial returns. Width means `(upper − lower)/(point prediction + 1)`; its median dollar width is {_money(main['median_interval_width_usd'])}. Temporal drift removes the usual exchangeability guarantee for the nominal 80% intervals.

![Temporal holdout performance](../figures/temporal_holdout.png)

## Candidate and comparable-price benchmarks

{_table(['Candidate', 'MAE/night', 'Median AE', 'Median APE', 'Within ±25%', '80% coverage', 'Relative width'], metric_rows)}

The two strong baselines use October-only comparable cells, falling back through a fixed hierarchy when a cell has fewer than 20 training listings or five hosts. One computes listing medians, the other medians of within-cell host medians. The primary HGB is trained on {summary['training_listings']:,} October long-stay listings from {summary['training_hosts']:,} hosts, with weights inverse to each host's eligible listing count. Host IDs are not features. The original listing-weighted candidate remains secondary even if a metric is numerically better.

The primary MAE is {_money(main['mae_usd'])}/night, median absolute error {_money(main['median_ae_usd'])}, median absolute percentage error {_percent(main['median_ape_pct'])}, and {_percent(main['within_25pct'])} of predictions are within 25% of the observed asking price. This is a screening benchmark with material individual uncertainty, not automatic price posting.

## Paired uncertainty and transit

{_table(['Contrast', 'Point estimate', '98.3333% paired-bootstrap interval'], contrast_rows)}

There are 3,000 paired resamples of the same primary hosts with seed 42. The individual 98.3333% percentile intervals allocate a 5% Bonferroni family across the two baseline improvements and the transit contrast. Positive baseline contrasts favor the primary model; positive transit changes mean higher error with transit. These are conditional empirical intervals.

{equiv_text} The transit point difference is {_percent(equivalence['estimate_pct'], 2)}, interval [{_number(equivalence['family_ci_lower_pct'])}%, {_number(equivalence['family_ci_upper_pct'])}%]. Neither result identifies a causal transit effect.

## Recurring market benchmark

{recurrence_en} The locked principal comparison is entire homes, one bedroom, declared capacity two, minimum exactly 30 nights and subway distance ≤500 m. Each borough/period requires at least 100 listings and 30 hosts. Brooklyn must be at least 10% lower in both 2026 periods and remain lower after excluding each borough's largest host.

{_table(['Period', 'Manhattan median', 'Brooklyn median', 'BK gap', 'Gap after largest-host exclusion', 'Status'], market_rows)}

![Fixed market benchmark](../figures/temporal_market_replication.png)

{_table(['Period', 'Borough', 'Largest-account listings', 'Account share', 'Median after exclusion', 'Median of host medians'], concentration_rows)}

October is historical context and does not count as a future replication. Market rows use all eligible listings in the fixed category, so listings and accounts can recur across periods; this is separate from the strict novel-host prediction test. Shared category filters do not match apartment floor area, quality, amenities or neighborhood composition. Capacity 2–4, private-room and 800 m rows remain labeled sensitivities in the source CSV and cannot rescue a failed principal comparison.

## Cohort and subgroup boundaries

{_table(['Cohort: primary model only', 'Listings', 'Hosts', 'MAE', 'Median APE', '80% coverage'], cohort_rows)}

The primary cohort selects one strict novel-host listing using the smallest SHA256 of `temporal-v1|` plus listing ID, independent of prices. Novelty checks use all prior raw IDs, including missing-price and ineligible records. “Unseen” means absent from those snapshots, not proven newly created on Airbnb. September does not recalibrate intervals or refit point models.

{_table(['Borough', 'Room type', 'n', 'Primary MAE', 'Median APE', '80% coverage', 'Diagnostic flags'], _subgroup_rows(subgroups))}

Subgroup flags are diagnostics, not extra post-hoc adoption gates. Rows with n<30 are shown but do not support reliable subgroup claims. Worse baseline comparisons or weak coverage are retained rather than dropping the affected group. Full model/baseline subgroup metrics are in `results/tables/temporal_holdout_subgroups.csv`.

## Selection, integrity and permitted use

{_table(['Snapshot', 'Raw rows', 'Missing/invalid price', 'Raw missing-price rate', 'Eligible long-stay rows', 'Eligible rows missing income'], quality_rows)}

Missing-price rates use all raw rows as the denominator. The observed positive-price population is not all NYC accommodation. Eligibility excludes recorded hotel/hostel/resort categories and minimum stays below 30 nights, with no price-tail trimming. Coordinates can be displaced by about 150 m; the station extract date is unknown. ACS 2019–2023 and TIGER/Line 2023 references remain frozen.

The intended use is historical long-stay asking-price screening and comparison using contemporaneous 2026 property/location information. It is not a forecast using only October information, a booked-price estimate, a property valuation, or evidence of demand, occupancy, revenue, legal eligibility or profit.

Source hashes: `results/temporal_source_manifest.json`. Prediction freeze: `results/temporal_prediction_freeze.json`. Complete decisions/metrics: `results/temporal_validation.json`. Primary candidate and both baselines were fixed before future scoring; June labels calibrate intervals only. No secondary candidate replaces a failed primary decision.
"""
    zh = f"""# 按锁定规则完成的时间外验证结论

**{chinese_decisions[summary['decision']]}** 主测试集为 2026 年 9 月的 **{summary['primary_test_hosts']:,} 个房东账号，各取一条房源**；这些账号和房源 ID 均未出现在 2025 年 10 月及 2026 年 6 月的全部原始记录中。6 月仅用于区间校准，样本为 {summary['calibration_hosts']:,} 个新账号。{sample_margin_zh}，不能据此声称所有地区、房型或未来时期都同样可靠。

验证方案在下载未来快照前已锁定，commit 为 `{PROTOCOL_COMMIT}`，见 [完整方案](TEMPORAL_EVALUATION_PROTOCOL.md)。这是有预设规则的回顾性留出验证，不是注册实验，也不是因果识别。

## 1. 模型检验得到什么

主模型是按账号平衡权重训练的直方图梯度提升模型，MAE 为 **{_money(main['mae_usd'])}/晚**；绝对误差中位数 **{_money(main['median_ae_usd'])}**，绝对百分比误差中位数 **{_percent(main['median_ape_pct'])}**。**{_percent(main['within_25pct'])}** 的点预测在挂牌价 ±25% 以内。名义 80% 区间的实际覆盖率是 **{_percent(main['coverage_80_pct'])}**，区间宽度中位数 **{_money(main['median_interval_width_usd'])}**。因此适用定位是有不确定性提示的价格筛选参考，需要人工查看具体房源。

{_table(['锁定标准', '实际结果', '要求', '判定'], _gate_rows(summary, main, contrasts, chinese=True))}

两个基线都只使用 10 月资料，在社区、房型、容量、卧室和最低入住要求等固定层级中找可比组；不足 20 条房源或 5 个账号时按锁定顺序回退。一个取挂牌中位数，另一个先取账号内中位数，再取账号间中位数。主模型按训练账号的房源数量反向加权，避免大账号凭房源数量主导训练；房东 ID 本身不是特征。

基线误差降幅及区间见上表。区间使用相同测试账号的配对重抽样 3,000 次，三个预设比较采用 Bonferroni 分配后的单个 98.3333% 区间。阈值是研究者预设的工程筛选标准，不是行业认证或真实商业收益证明。

![时间外验证](../figures/temporal_holdout.png)

## 2. 整套房区域报价基准是否重复出现

**{recurrence_zh}** 条件固定为整套、一卧、标注可住恰好 2 人、最低入住恰好 30 晚、地铁直线距离不超过 500 米。每区每期至少需要 100 条房源和 30 个账号；两个 2026 快照中 Brooklyn 中位价都至少低 10%，且移除各区最大账号后仍较低，才算重复成立。

{_table(['快照', 'Manhattan 中位价及样本', 'Brooklyn 中位价及样本', 'Brooklyn 相对差价', '移除最大账号后的差价', '锁定判定'], market_rows_zh)}

![固定区域报价基准](../figures/temporal_market_replication.png)

这是固定住宿类别在多个快照中的历史报价规律，能支持分层比较。共同条件不等于装修、面积、具体社区或房屋质量完全相同；不能解释成相同房屋换区后的因果差价。市场表使用该类别的全部合格房源，账号或房源可跨期重复，因此不能把它与新账号预测测试当成同一种独立性。等账号权重及最大账号剔除值均在英文报告和结果表中保留。

## 3. 地铁结论需要收紧

加入地铁后的相对 MAE 变化为 **{_percent(equivalence['estimate_pct'], 2)}**，预设区间为 **[{_number(equivalence['family_ci_lower_pct'])}%, {_number(equivalence['family_ci_upper_pct'])}%]**。{equiv_zh} 该比较只讨论预测表现，不能证明地铁对价格有无因果影响。

## 4. 子组失败和范围限制

{_table(['地区', '房型', 'n', '主模型 MAE', '误差百分比中位数', '80% 覆盖率', '诊断标记'], _subgroup_rows(subgroups, chinese=True))}

这些标记是透明呈现的子组诊断，不是看过结果后新增的整体采用门槛。少于 30 条的子组不能支持稳定的地区/房型结论。所有主、次模型与基线指标都在英文报告和 CSV 中保留，不能用表现更好的次模型或更容易的样本替换锁定主结果。

数据只覆盖有正挂牌价的非酒店长住样本。缺价、物业分类、坐标约 150 米扰动、未知日期的地铁快照及滞后的 Census 数据都限制解释。预测使用后期房源的同期属性，不是只凭 2025 年信息预测未来成交价。6 月只校准区间，9 月没有重新校准；跨期变化使名义 80% 区间不具备可直接宣称的分布无关未来覆盖保证。

## 简历中可以准确使用的表述

基于 NYC Airbnb、MTA 与 Census 数据构建长住挂牌价筛选模型及分层区域基准；在下载留出数据前锁定协议，按未见账号抽样进行 2026 年时间外验证，比较两个训练期可比房源基线，并报告预测区间、账号构成敏感性及子组失败。

若采用具体数字，应同时注明 **{summary['primary_test_hosts']:,} 个未见账号的留出样本、MAE {_money(main['mae_usd'])}，以及上表的锁定判定**。结果不代表营收、入住率、自动定价收益或因果证明。完整方法、所有候选模型、选择偏差与输入哈希见 [英文验证报告](TEMPORAL_VALIDATION.md)。
"""
    english_path, chinese_path = docs / 'TEMPORAL_VALIDATION.md', docs / 'VERIFIED_CONCLUSIONS_ZH.md'
    english_path.write_text(english, encoding='utf-8')
    chinese_path.write_text(zh, encoding='utf-8')
    return {'english_report': str(english_path), 'chinese_report': str(chinese_path),
            'holdout_figure': str(figures / 'temporal_holdout.png'),
            'market_figure': str(figures / 'temporal_market_replication.png')}
