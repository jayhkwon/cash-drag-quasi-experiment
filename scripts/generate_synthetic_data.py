#!/usr/bin/env python3
"""Generate synthetic data for a cash-drag nudges case study.

Outputs:
- data/accounts.csv
- data/account_month.csv
- data/nudge_campaigns.csv
- data/did_ready.csv

Calibration notes (public-facing):
- Balances are lognormal to mimic right-skewed retirement account distributions.
- Investment propensity declines with months since rollover to reflect cash drag.
- Nudge assignment is more likely for higher engagement and larger balances
  to reflect targeted outreach and selection bias.
- Exposure is lower than assignment to reflect non-compliance (opens/views).
- A small contamination rate allows occasional exposure in the control group.
- Seasonality and missingness are added to reflect operational data realities.

These settings are designed to mirror realistic patterns, not proprietary data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class Config:
    seed: int = 42
    n_accounts: int = 50_000
    n_months: int = 36
    rollover_start: int = 1
    rollover_end: int = 12
    close_rate: float = 0.001
    contam_rate: float = 0.03
    missing_engagement_rate: float = 0.05
    missing_market_rate: float = 0.02
    label_noise_rate: float = 0.01


def build_accounts(rng: np.random.Generator, cfg: Config) -> pd.DataFrame:
    account_id = np.arange(1, cfg.n_accounts + 1)
    rollover_month = rng.integers(cfg.rollover_start, cfg.rollover_end + 1, size=cfg.n_accounts)

    baseline_balance = rng.lognormal(mean=10.3, sigma=0.8, size=cfg.n_accounts)
    baseline_balance = np.clip(baseline_balance, 1_000, 500_000)

    age_band = rng.choice(["18-34", "35-49", "50-64", "65+"], p=[0.20, 0.30, 0.30, 0.20], size=cfg.n_accounts)
    tenure_band = rng.choice(["0-2y", "3-5y", "6y+"], p=[0.35, 0.35, 0.30], size=cfg.n_accounts)
    risk_tolerance = rng.choice(["low", "medium", "high"], p=[0.40, 0.40, 0.20], size=cfg.n_accounts)

    engagement = rng.beta(2, 5, size=cfg.n_accounts)
    advisor_flag = rng.binomial(1, 0.25, size=cfg.n_accounts)

    regions = ["North", "South", "East", "West"]
    region = rng.choice(regions, size=cfg.n_accounts)

    contactable_email = rng.binomial(1, 0.85, size=cfg.n_accounts)
    contactable_sms = rng.binomial(1, 0.55, size=cfg.n_accounts)

    preferred_channel = rng.choice(["email", "sms", "inapp"], p=[0.50, 0.15, 0.35], size=cfg.n_accounts)

    account_fe = rng.normal(0, 0.5, size=cfg.n_accounts)

    return pd.DataFrame(
        {
            "account_id": account_id,
            "rollover_month": rollover_month,
            "baseline_balance": baseline_balance,
            "age_band": age_band,
            "tenure_band": tenure_band,
            "risk_tolerance": risk_tolerance,
            "digital_engagement_score": engagement,
            "advisor_flag": advisor_flag,
            "region": region,
            "contactable_email": contactable_email,
            "contactable_sms": contactable_sms,
            "preferred_channel": preferred_channel,
            "account_fe": account_fe,
        }
    )


def build_campaigns(cfg: Config) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    regions = ["North", "South", "East", "West"]
    region_start = {"North": 4, "South": 7, "East": 10, "West": 13}
    channel_offset = {"email": 0, "sms": 1, "inapp": 2}
    channel_intensity = {"email": 1, "sms": 2, "inapp": 1}

    rows = []
    wave_id = 1
    map_by_channel: dict[str, dict[str, int]] = {"email": {}, "sms": {}, "inapp": {}}
    map_end_by_channel: dict[str, dict[str, int]] = {"email": {}, "sms": {}, "inapp": {}}
    map_wave_by_channel: dict[str, dict[str, int]] = {"email": {}, "sms": {}, "inapp": {}}

    for region in regions:
        for channel in ["email", "sms", "inapp"]:
            start = region_start[region] + channel_offset[channel]
            end = min(start + 12, cfg.n_months)
            intensity = channel_intensity[channel]

            rows.append(
                {
                    "nudge_wave_id": wave_id,
                    "channel": channel,
                    "region": region,
                    "start_month": start,
                    "end_month": end,
                    "eligibility_rule": "months_since_rollover in [1,6] and not invested",
                    "intensity": intensity,
                }
            )

            map_by_channel[channel][region] = start
            map_end_by_channel[channel][region] = end
            map_wave_by_channel[channel][region] = wave_id
            wave_id += 1

    campaigns = pd.DataFrame(rows)
    maps = {
        "start": map_by_channel,
        "end": map_end_by_channel,
        "wave": map_wave_by_channel,
    }
    return campaigns, maps


def generate_panel(accounts: pd.DataFrame, campaigns_map: dict[str, dict[str, dict[str, int]]], cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)
    n = len(accounts)

    rollover_month = accounts["rollover_month"].to_numpy()
    balance = accounts["baseline_balance"].to_numpy()
    engagement = accounts["digital_engagement_score"].to_numpy()
    advisor_flag = accounts["advisor_flag"].to_numpy()
    region = accounts["region"].to_numpy()

    contactable_email = accounts["contactable_email"].to_numpy().astype(bool)
    contactable_sms = accounts["contactable_sms"].to_numpy().astype(bool)

    account_fe = accounts["account_fe"].to_numpy()

    # Campaign mapping per account
    email_start = np.vectorize(campaigns_map["start"]["email"].get)(region)
    sms_start = np.vectorize(campaigns_map["start"]["sms"].get)(region)
    inapp_start = np.vectorize(campaigns_map["start"]["inapp"].get)(region)

    email_end = np.vectorize(campaigns_map["end"]["email"].get)(region)
    sms_end = np.vectorize(campaigns_map["end"]["sms"].get)(region)
    inapp_end = np.vectorize(campaigns_map["end"]["inapp"].get)(region)

    email_wave = np.vectorize(campaigns_map["wave"]["email"].get)(region)
    sms_wave = np.vectorize(campaigns_map["wave"]["sms"].get)(region)
    inapp_wave = np.vectorize(campaigns_map["wave"]["inapp"].get)(region)

    # Monthly macro factors
    market_return = rng.normal(0.004, 0.03, size=cfg.n_months)
    seasonality_template = np.array([0.20, 0.10, 0.05, 0.00, -0.05, 0.00, 0.05, 0.05, 0.00, 0.05, 0.10, 0.20])
    seasonality = np.tile(seasonality_template, int(np.ceil(cfg.n_months / 12)))[: cfg.n_months]
    seasonality = seasonality * 0.10

    # State
    invested_status = np.zeros(n, dtype=bool)
    account_closed = np.zeros(n, dtype=bool)
    first_exposure_month = np.full(n, -1, dtype=int)

    rows = []

    for m in range(1, cfg.n_months + 1):
        msr = m - rollover_month
        active = msr >= 0
        eligible = active & (msr >= 1) & (msr <= 6) & (~invested_status) & (~account_closed)

        p_assign = 0.05 + 0.10 * engagement + 0.05 * (balance > 50_000)

        email_active = (m >= email_start) & (m <= email_end)
        sms_active = (m >= sms_start) & (m <= sms_end)
        inapp_active = (m >= inapp_start) & (m <= inapp_end)

        email_assign = eligible & contactable_email & email_active & (rng.random(n) < p_assign)
        sms_assign = eligible & contactable_sms & sms_active & (rng.random(n) < p_assign)
        inapp_assign = eligible & inapp_active & (rng.random(n) < p_assign)

        email_open_prob = np.clip(0.35 + 0.25 * engagement, 0.0, 0.95)
        sms_open_prob = np.clip(0.55 + 0.20 * engagement, 0.0, 0.98)
        inapp_open_prob = np.clip(0.45 + 0.30 * engagement, 0.0, 0.98)

        exposure_email = email_assign & (rng.random(n) < email_open_prob)
        exposure_sms = sms_assign & (rng.random(n) < sms_open_prob)
        exposure_inapp = inapp_assign & (rng.random(n) < inapp_open_prob)

        # Contamination (unexpected exposure)
        exposure_email |= eligible & (~email_assign) & (rng.random(n) < cfg.contam_rate)
        exposure_sms |= eligible & (~sms_assign) & (rng.random(n) < cfg.contam_rate)
        exposure_inapp |= eligible & (~inapp_assign) & (rng.random(n) < cfg.contam_rate)

        n_exposed = exposure_email.astype(int) + exposure_sms.astype(int) + exposure_inapp.astype(int)

        treat_effect = (
            0.35 * exposure_email.astype(float)
            + 0.45 * exposure_sms.astype(float)
            + 0.30 * exposure_inapp.astype(float)
            - 0.15 * np.maximum(n_exposed - 1, 0)
        )

        msr_for_model = np.where(active, msr, 0)
        logit = (
            -4.2
            + 0.35 * np.log(balance)
            + 0.60 * engagement
            - 0.10 * msr_for_model
            + 0.20 * market_return[m - 1]
            + 0.15 * advisor_flag
            + seasonality[m - 1]
            + account_fe
        )

        p = logistic(logit + treat_effect)
        p = p * active * (~invested_status) * (~account_closed)

        invest_true = rng.random(n) < p

        # Label noise
        invested_flag = invest_true.copy()
        noise_mask = rng.random(n) < cfg.label_noise_rate
        invested_flag = np.where(noise_mask, ~invested_flag, invested_flag)

        allocation_rate = rng.beta(5, 2, size=n)
        invest_amount = np.where(invest_true, balance * allocation_rate, 0.0)

        # Update first exposure month (event time)
        newly_exposed = (n_exposed > 0) & (first_exposure_month < 0)
        first_exposure_month = np.where(newly_exposed, m, first_exposure_month)

        # Update invested status
        invested_status = invested_status | invest_true

        # Account closure after investment decision
        closed_now = (~account_closed) & (rng.random(n) < cfg.close_rate)
        account_closed = account_closed | closed_now

        # Missingness
        missing_engagement = rng.random(n) < cfg.missing_engagement_rate
        engagement_obs = np.where(missing_engagement, np.nan, engagement)

        missing_market = rng.random(n) < cfg.missing_market_rate
        market_return_obs = np.where(missing_market, np.nan, market_return[m - 1])

        event_time = np.where(first_exposure_month >= 0, m - first_exposure_month, np.nan)
        treated = (first_exposure_month >= 0).astype(int)
        post = ((first_exposure_month >= 0) & (m >= first_exposure_month)).astype(int)

        month_df = pd.DataFrame(
            {
                "account_id": accounts["account_id"].to_numpy(),
                "calendar_month": m,
                "months_since_rollover": msr,
                "active_flag": active.astype(int),
                "eligible_flag": eligible.astype(int),
                "market_return": market_return[m - 1],
                "seasonality_index": seasonality[m - 1],
                "nudge_email": email_assign.astype(int),
                "nudge_sms": sms_assign.astype(int),
                "nudge_inapp": inapp_assign.astype(int),
                "exposure_email": exposure_email.astype(int),
                "exposure_sms": exposure_sms.astype(int),
                "exposure_inapp": exposure_inapp.astype(int),
                "email_wave_id": email_wave,
                "sms_wave_id": sms_wave,
                "inapp_wave_id": inapp_wave,
                "treatment_any": (n_exposed > 0).astype(int),
                "treatment_intensity": n_exposed,
                "first_exposure_month": first_exposure_month,
                "event_time": event_time,
                "treated": treated,
                "post": post,
                "invested_true": invest_true.astype(int),
                "invested_flag": invested_flag.astype(int),
                "invested_status": invested_status.astype(int),
                "invest_amount": invest_amount,
                "account_closed": account_closed.astype(int),
                "missing_engagement": missing_engagement.astype(int),
                "engagement_obs": engagement_obs,
                "missing_market_return": missing_market.astype(int),
                "market_return_obs": market_return_obs,
            }
        )

        rows.append(month_df)

    panel = pd.concat(rows, ignore_index=True)
    return panel


def build_did_ready(panel: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "account_id",
        "calendar_month",
        "months_since_rollover",
        "treated",
        "post",
        "event_time",
        "treatment_any",
        "treatment_intensity",
        "invested_flag",
        "invest_amount",
        "active_flag",
        "eligible_flag",
    ]
    return panel[cols].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accounts", type=int, default=50_000)
    parser.add_argument("--months", type=int, default=36)
    args = parser.parse_args()

    cfg = Config(seed=args.seed, n_accounts=args.accounts, n_months=args.months)
    rng = np.random.default_rng(cfg.seed)

    accounts = build_accounts(rng, cfg)
    campaigns, maps = build_campaigns(cfg)
    panel = generate_panel(accounts, maps, cfg)
    did_ready = build_did_ready(panel)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    accounts.to_csv(outdir / "accounts.csv", index=False)
    panel.to_csv(outdir / "account_month.csv", index=False)
    campaigns.to_csv(outdir / "nudge_campaigns.csv", index=False)
    did_ready.to_csv(outdir / "did_ready.csv", index=False)

    print("Wrote datasets to", outdir)


if __name__ == "__main__":
    main()
