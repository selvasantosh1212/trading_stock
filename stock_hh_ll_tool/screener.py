from typing import Optional

import pandas as pd

from smc_validator.signal.htf_bias import combine_bias
from smc_validator.structure.bos_choch import compute_structure
from smc_validator.structure.swings import detect_fractal_pivots
from stock_hh_ll_tool.data import fetch_ohlc
from stock_hh_ll_tool.gaps import detect_overnight_gaps
from stock_hh_ll_tool.targets import find_next_target


def compute_trend(df: pd.DataFrame, left: int, right: int, close_only: bool = True) -> pd.DataFrame:
    pivots = detect_fractal_pivots(df, left, right)
    return compute_structure(df, pivots, close_only=close_only)


def evaluate_daily_signal(daily_df: pd.DataFrame, cfg: dict) -> dict:
    """Pure logic (no network) — the swing-trade signal: did the LATEST
    daily bar produce a BOS/CHoCH in either direction (a confirmed Higher
    High or Lower Low), and was there a notable overnight gap alongside it.
    When a break fired today, also looks up the next older unbroken
    swing level in that direction (see targets.find_next_target — useful
    context, not a validated prediction). Separated from data-fetching so
    this is directly unit-testable with synthetic OHLC.
    """
    left = cfg["daily_structure"]["external"]["left_bars"]
    right = cfg["daily_structure"]["external"]["right_bars"]
    close_only = cfg["daily_structure"]["close_only_break"]
    structure = compute_trend(daily_df, left, right, close_only)

    today_event = structure["event"].iloc[-1]
    today_direction = structure["event_direction"].iloc[-1]
    broken_today = today_event in ("BOS", "CHoCH")
    hh_broken_today = broken_today and today_direction == "bullish"
    ll_broken_today = broken_today and today_direction == "bearish"

    gaps = detect_overnight_gaps(daily_df, cfg["gap"]["min_gap_pct"])
    gap_pct_today = gaps["gap_pct"].iloc[-1]

    next_target = None
    if hh_broken_today or ll_broken_today:
        next_target = find_next_target(daily_df, structure, today_direction)

    return {
        "date": str(daily_df.index[-1].date()),
        "close": float(daily_df["close"].iloc[-1]),
        "hh_broken_today": bool(hh_broken_today),
        "ll_broken_today": bool(ll_broken_today),
        "structure_event": today_event,
        "structure_event_direction": today_direction,
        "daily_trend": structure["trend"].iloc[-1],
        "gapped_up_today": bool(gaps["gap_up"].iloc[-1]),
        "gapped_down_today": bool(gaps["gap_down"].iloc[-1]),
        "gap_pct": float(gap_pct_today) if pd.notna(gap_pct_today) else None,
        "next_target": next_target,
    }


def evaluate_intraday_bias(
    trend_by_timeframe: dict[str, Optional[str]], cfg: dict, expected_direction: str = "bullish"
) -> dict:
    """Pure logic — combines each intraday timeframe's latest trend value
    into a bias-agreement check against `expected_direction` (whichever
    direction the daily break just fired in), reusing the exact same
    `combine_bias` function from the forex project (same fractal,
    per-timeframe structure principle, different market).

    NOTE: this `intraday_confirmation` feature predates, and is NOT
    validated the same way as, `entry_exit.py`'s Weekly/Daily strategy (see
    reports/stock_entry_exit_strategy_investigation.md) — it's an earlier,
    unvalidated heuristic kept for now as a secondary display-only signal,
    not something to treat with the same confidence.
    """
    icfg = cfg["intraday_confirmation"]
    bias_tfs = [tf for tf in icfg["bias_timeframes"] if tf in trend_by_timeframe]

    bias_series = {tf: pd.Series([trend_by_timeframe[tf]]) for tf in bias_tfs}
    combined = combine_bias(bias_series) if bias_series else pd.Series([None])

    return {
        "per_timeframe_trend": dict(trend_by_timeframe),
        "bias_timeframes_agree": bool(combined.iloc[0] == expected_direction),
    }


def screen_symbol(symbol: str, cfg: dict, daily: pd.DataFrame | None = None) -> dict:
    """Fetches data (unless `daily` is already provided by the caller, to
    avoid double-fetching when the runner script also needs the same daily
    bars for the entry/exit check) and runs the full check for one symbol.
    Never raises on a single symbol's data problem (a delisted ticker, a
    Yahoo hiccup, a typo) — returns an "error" key instead, so one bad
    ticker in a watchlist doesn't abort the whole run.
    """
    if daily is None:
        try:
            daily = fetch_ohlc(symbol, "1d")
        except Exception as e:
            return {"symbol": symbol, "error": str(e)}

    result = {"symbol": symbol, **evaluate_daily_signal(daily, cfg), "intraday_bias": None}

    if result["hh_broken_today"] or result["ll_broken_today"]:
        icfg = cfg["intraday_confirmation"]
        left, right = icfg["external"]["left_bars"], icfg["external"]["right_bars"]
        trend_by_timeframe: dict[str, Optional[str]] = {}
        # only fetch bias_timeframes — entry_timeframes aren't consumed by
        # evaluate_intraday_bias, so fetching them was pure wasted network
        # I/O (this ad-hoc feature's finer-grained entry timing is now
        # handled properly by entry_exit.py instead)
        for tf in icfg["bias_timeframes"]:
            try:
                bars = fetch_ohlc(symbol, tf)
                trend_by_timeframe[tf] = compute_trend(bars, left, right)["trend"].iloc[-1]
            except Exception:
                continue
        result["intraday_bias"] = evaluate_intraday_bias(
            trend_by_timeframe, cfg, expected_direction=result["structure_event_direction"]
        )

    return result
