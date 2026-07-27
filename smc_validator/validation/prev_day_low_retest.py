import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from smc_validator.data_ingestion.yahoo_daily import load_daily

INTERPRETATIONS = {
    "close_below_prior_low": "Interpretation A: close < prior day's low (any candle color)",
    "red_candle_close_below_prior_low": "Interpretation B: red candle AND close < prior day's low",
}


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index().copy()
    df["prior_low"] = df["low"].shift(1)
    df["next_low"] = df["low"].shift(-1)
    df["close_below_prior_low"] = df["close"] < df["prior_low"]
    df["red_candle_close_below_prior_low"] = (df["close"] < df["open"]) & df["close_below_prior_low"]
    # outcome uses next_low vs prior_low (NOT vs the current day's own low) —
    # this is testing whether the NEXT day revisits the SAME level that was
    # broken, per the source claim.
    df["touched_prior_low_next_day"] = df["next_low"] <= df["prior_low"]
    return df.dropna(subset=["prior_low", "next_low"])


def _rate_with_ci(subset: pd.DataFrame, outcome_col: str) -> dict:
    n = len(subset)
    if n == 0:
        return {"n": 0, "successes": 0, "rate": None, "ci_low": None, "ci_high": None}
    successes = int(subset[outcome_col].sum())
    rate = successes / n
    ci_low, ci_high = proportion_confint(successes, n, method="wilson")
    return {"n": n, "successes": successes, "rate": rate, "ci_low": ci_low, "ci_high": ci_high}


def run_validation(train_test_split_fraction: float = 0.7) -> dict:
    """Pre-registered, chronological (never random) out-of-sample test of the
    video's claim: after a bearish close below the previous day's low, there's
    a ~75% chance the next day also trades down to that same level. Reports
    both stated interpretations, in-sample vs out-of-sample, and — critically
    — the unconditional baseline touch rate, since prior lows get revisited
    often from ordinary volatility and the 75% figure is meaningless without
    that comparison.
    """
    df = _prepare(load_daily())

    split_idx = int(len(df) * train_test_split_fraction)
    in_sample, out_of_sample = df.iloc[:split_idx], df.iloc[split_idx:]

    results: dict = {
        "n_total_days": len(df),
        "date_range": (str(df.index[0].date()), str(df.index[-1].date())),
        "unconditional_baseline": _rate_with_ci(df, "touched_prior_low_next_day"),
    }

    for col in INTERPRETATIONS:
        results[col] = {
            "in_sample": _rate_with_ci(in_sample[in_sample[col]], "touched_prior_low_next_day"),
            "out_of_sample": _rate_with_ci(out_of_sample[out_of_sample[col]], "touched_prior_low_next_day"),
            "full_sample": _rate_with_ci(df[df[col]], "touched_prior_low_next_day"),
        }

    return results


def _fmt(p: dict) -> str:
    if p["n"] == 0:
        return "no qualifying days"
    return (
        f"{p['rate']:.1%} (n={p['n']}, successes={p['successes']}, "
        f"95% CI [{p['ci_low']:.1%}, {p['ci_high']:.1%}])"
    )


def format_report(results: dict) -> str:
    lines = [
        "# Previous-Day-Low Retest Validation",
        "",
        f"Data: {results['n_total_days']} daily bars, "
        f"{results['date_range'][0]} to {results['date_range'][1]} (EURUSD, Yahoo Finance daily bars).",
        "",
    ]

    baseline = results["unconditional_baseline"]
    lines.append(f"**Unconditional baseline** (any day, regardless of setup): {_fmt(baseline)}")
    lines.append("")

    for col, title in INTERPRETATIONS.items():
        r = results[col]
        lines.append(f"## {title}")
        lines.append(f"- In-sample (first 70%, chronological): {_fmt(r['in_sample'])}")
        lines.append(f"- Out-of-sample (last 30%, chronological): {_fmt(r['out_of_sample'])}")
        lines.append(f"- Full sample: {_fmt(r['full_sample'])}")
        lines.append("")

    lines.append("## Verdict (out-of-sample vs unconditional baseline)")
    for col, title in INTERPRETATIONS.items():
        oos = results[col]["out_of_sample"]
        if oos["n"] == 0:
            lines.append(f"- {title}: insufficient out-of-sample data.")
            continue
        lift = oos["rate"] - baseline["rate"]
        distinguishable = oos["ci_low"] > baseline["ci_high"]
        verdict = "meaningfully above baseline" if distinguishable else "NOT clearly distinguishable from baseline (CIs overlap)"
        lines.append(
            f"- {title}: out-of-sample {oos['rate']:.1%} vs baseline {baseline['rate']:.1%} "
            f"(lift {lift:+.1%}) — **{verdict}**."
        )
    lines.append("")
    lines.append(
        "Note: Yahoo Finance's daily FX bars are not guaranteed to align exactly with the "
        "17:00 New York forex-day convention; treat results as directionally informative, "
        "not a millisecond-precise replication of the original claim's methodology."
    )

    return "\n".join(lines)
