# cash_drag: Synthetic Causal Inference Case Study

This repository studies a business problem: can multi-channel nudges reduce rollover IRA cash drag, and how robust are conclusions under realistic operational frictions.

The project is synthetic by design, but intentionally not "clean toy data." It includes targeting, non-compliance, overlap, trend, and missingness behaviors that make identification harder and closer to production reality.

## Executive Summary

- **Task:** estimate whether multi-channel nudges increase investing behavior after rollover (reduce cash drag).
- **Primary design:** staggered-adoption DiD with account and month fixed effects, plus event-study diagnostics.
- **Robustness stack:** eventually-treated FE view, cohort/event-time aggregates (CS/SA-style), IPW, doubly robust ATE, and fuzzy RD sensitivity checks.
- **Directional conclusion:** effect is consistently positive across DiD-family and account-level robustness estimators.
- **Caution:** pre-trend diagnostics are mixed in the full sample; decision framing is guardrail-based, not “single-number certainty.”
- **Decision objective:** scale nudges only when incremental lift is economically material and diagnostics remain within policy limits.

## Results at a Glance

Validated on `2026-02-10` with default synthetic generator settings (`seed=42`, `50k` accounts, `36` months).

| Estimator / View | Target | Estimate | Uncertainty / Diagnostic | Decision Interpretation |
|---|---|---:|---|---|
| TWFE-style DiD | Monthly invest probability | +0.1638 | Clustered SE ~0.001 | Strong positive directional signal |
| Eventually-treated FE | Monthly invest probability | +0.1607 | Clustered SE ~0.001 | Similar magnitude within treated cohorts |
| CS-style post ATT (`e>=0`) | Monthly invest probability | +0.1883 | Pre mean (`e<=-2`) = -0.0858 | Positive post effect; interpret with pre-period caution |
| SA-style post ATT (`e>=0`) | Monthly invest probability | +0.1883 | Same event-time aggregation family | Supports directional consistency |
| IPW ATE | Invest within 12 months | +0.3739 | 95% CI [0.3540, 0.3949] | Positive account-level effect |
| Doubly robust ATE | Invest within 12 months | +0.3738 | 95% CI [0.3664, 0.3815] | Robust positive effect |
| Fuzzy RD (`bw=2`) | Local cutoff effect | +0.0635 | 95% CI [-0.1885, 0.3223] | Unstable/local only; sensitivity evidence, not primary |

## Explainability + Decisioning Framing

“Explainability” in this causal setting means clarity on **where** effects appear and **when** they are credible:

1. Event-study decomposition explains timing and pre/post dynamics.
2. Segment heterogeneity (low/mid/high balance lift) explains where uplift is strongest.
3. Balance diagnostics (`|SMD|`) explain whether weighting corrected comparability.
4. RD sensitivity explains local-instability risk near eligibility boundaries.

What this means for a risk or operations team:

1. Treat DiD-family uplift as decision-grade directional evidence.
2. Use robustness agreement (TWFE + eventually-treated + IPW/DR) as confidence support.
3. Keep rollout tied to explicit governance triggers when diagnostics deteriorate.

## Scope

- Deep EDA and data credibility checks before modeling.
- Staggered-adoption DiD and event-study diagnostics.
- Modern DiD-style robustness views (eventually-treated FE, cohort/event-time aggregates).
- Account-level causal robustness (IPW and doubly robust ATE).
- Local sensitivity checks via fuzzy RD, including placebo/specification checks and cohort-stratified and local-randomization diagnostics.

## Reproducibility

Data is generated locally and excluded from version control.

Suggested setup (relative paths only):

```bash
cd cash_drag
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas scipy statsmodels matplotlib seaborn jupyter
python scripts/generate_synthetic_data.py --outdir data
```

Minimal data-generation command:

```bash
python scripts/generate_synthetic_data.py --outdir data
```

Optional arguments:

- `--seed` default `42`
- `--accounts` default `50000`
- `--months` default `36`

Recommended notebook order:

1. `eda_deep_dive.ipynb`
2. `did_event_study.ipynb`
3. `other_models.ipynb`
4. `rd_identification.ipynb`

## How This Would Be Used in Production

1. Define a monthly scaling policy with explicit thresholds:
   - minimum holdout-adjusted lift,
   - minimum quality/fairness constraints,
   - downside guardrails for key segments.
2. Set rollout state each cycle: `scale`, `maintain`, or `pause`.
3. Keep a persistent holdout slice to validate drift-adjusted incremental lift.
4. Monitor operational diagnostics:
   - score/treatment/exposure drift,
   - segment-level degradation,
   - compliance and overlap behavior changes.
5. Recalibrate policy thresholds quarterly as economics and risk appetite shift.

## Tracked vs Ignored

- Tracked: notebooks, scripts, `README.md`, and `figures/`.
- Ignored: generated `data/`.

## Data-Generation Design

The synthetic generator deliberately includes these frictions:

- Targeted assignment into nudge treatment.
- Assignment-to-exposure non-compliance.
- Simultaneous/overlapping nudge channels and waves.
- Contamination and seasonality.
- Missingness in selected variables.
- Light measurement noise in observed investment labels.

Use this repository for methodology and decision-process testing, not as a reconstruction of any proprietary production dataset.

## EDA Coverage

`eda_deep_dive.ipynb` is model-agnostic and includes:

- Panel observability and completeness checks.
- Missingness profile and data integrity checks.
- Segment composition and baseline context.
- Eligibility, treatment, and investment funnel diagnostics.
- Monthly and lifecycle trend plots.
- Early-vs-late cohort drift via standardized mean differences.
- Correlation and segment-level outcome summaries.

Chart labels are business-readable (not raw variable names).

## Modeling Stack

Primary DiD family:

- TWFE-style staggered DiD with account and month fixed effects.
- Event-study around first exposure.
- Eventually-treated FE/no-never-treated view for within-target-population interpretation.
- Cohort/event-time aggregate robustness inspired by Callaway-Sant'Anna and Sun-Abraham logic.

Account-level robustness:

- IPW ATE on `invest_within_12_months`.
- Doubly robust ATE.
- Weighted/unweighted balance diagnostics with standardized mean differences.
- Bootstrap confidence intervals.

Local design sensitivity:

- Fuzzy RD at the eligibility boundary (`msr=6.5`) with bandwidth/specification/placebo checks.
- Cohort-stratified RD checks.
- Local-randomization diagnostics.

## Latest Results Snapshot

Validated on `2026-02-10` with default generator settings (`seed=42`, `50k` accounts, `36` months).

DiD-family outputs on monthly `invested_flag` scale:

- TWFE-style DiD coefficient: `+0.1638` (clustered SE about `0.001`).
- Eventually-treated FE coefficient: `+0.1607` (clustered SE about `0.001`).
- Event-study lead-joint p-value, full sample: `0.000000`.
- Event-study lead-joint p-value, eventually-treated cohorts: `0.216435`.
- Segment lift (same monthly DiD estimand):
- Low balance: `15.181 pp` (95% CI `14.837` to `15.524`).
- Mid balance: `16.663 pp` (95% CI `16.252` to `17.073`).
- High balance: `17.416 pp` (95% CI `16.937` to `17.896`).
- CS-style post ATT (`e>=0`): `+0.1883`; CS-style pre mean (`e<=-2`): `-0.0858`.
- SA-style post ATT (`e>=0`): `+0.1883`.

Account-level robustness on `invest_within_12_months`:

- IPW ATE: `+0.3739`.
- Doubly robust ATE: `+0.3738`.
- IPW bootstrap 95% CI: `[0.3540, 0.3949]`.
- DR bootstrap 95% CI: `[0.3664, 0.3815]`.
- Balance summary (`|SMD|`) improved:
- Mean: `0.048906` to `0.002609`.
- Max: `0.226497` to `0.005981`.
- Share below `0.1`: `0.823529` to `1.000000`.

Fuzzy RD local sensitivity:

- Main local-linear fuzzy RD (`bw=2`): Wald estimate `0.063505`, 95% CI `[-0.18851, 0.322332]`.
- Bandwidth/specification behavior is unstable (for example, median Wald in the quick bandwidth table is `-2.087084`).
- Interpret RD results here as local sensitivity diagnostics, not as a primary decision estimator.

## Data Credibility Snapshot

From the current generated panel:

- Accounts: `50,000`.
- Account-month rows: `1,800,000`.
- Coverage per account: median `36` months, IQR `36` to `36`.
- Ever eligible share: `0.58452`.
- Ever nudged share: `0.17276`.
- Ever invested share: `0.97692`.
- Top missingness: `event_time` at `0.867294` (expected because many accounts are never exposed).

## Interpreting Effects Correctly

Not all estimators target the same estimand:

- Monthly DiD/event-study estimates are on repeated monthly investment action.
- IPW/DR estimates are on account-level investment within 12 months.
- RD estimates are local to a cutoff neighborhood and highly bandwidth-sensitive in this synthetic setup.

Do not directly compare magnitudes across these targets without restating the estimand.

## Practical Modeling Note on Controls

In account FE plus month FE DiD:

- Time-invariant account attributes are absorbed by account fixed effects.
- Common monthly shocks are absorbed by month fixed effects.
- Post-treatment controls can induce bias.

The priority is careful pre-treatment adjustment and triangulation across estimators, not "more controls" by default.

## Conservative Claim Boundary

This project supports a decision-process claim:

- Directional signal is broadly positive across multiple methods.
- Magnitude is method- and estimand-dependent.
- Residual confounding risk remains explicit.
- Business use should be phased and guardrail-based rather than framed as one universally causal number.

## Core Files

- `scripts/generate_synthetic_data.py`
- `eda_deep_dive.ipynb`
- `did_event_study.ipynb`
- `other_models.ipynb`
- `rd_identification.ipynb`
- `data/accounts.csv` generated locally
- `data/account_month.csv` generated locally
- `data/did_ready.csv` generated locally
- `data/nudge_campaigns.csv` generated locally
