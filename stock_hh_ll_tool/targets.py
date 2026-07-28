import pandas as pd

from smc_validator.liquidity.daily_weekly import previous_period_levels


def find_next_target(df: pd.DataFrame, structure: pd.DataFrame, direction: str) -> dict:
    """After a bullish break (HH confirmed), the 'next target' is the
    nearest OLDER, still-unbroken swing high above the current price — the
    next visible resistance level on the chart. Symmetric for bearish
    breaks (nearest older unbroken swing low below current price, as the
    next support).

    Empirically checked, not just asserted (see reports/stock_target_investigation.md):
    this target does NOT show a statistically distinctive hit-rate
    advantage over an "any random day, same target definition" baseline
    across 5 major NSE stocks' 10-year history at 10/20/40-day horizons —
    the hit rate is similarly high whether or not a breakout just happened,
    because the target is by construction the nearest such level. Treat
    this as useful context (the next resistance/support to watch), not a
    validated breakout-specific prediction.
    """
    n = len(df)
    current_price = float(df["close"].iloc[-1])

    if direction == "bullish":
        idx_col, price_col = "swing_high_idx", df["high"].to_numpy()
    elif direction == "bearish":
        idx_col, price_col = "swing_low_idx", df["low"].to_numpy()
    else:
        raise ValueError(f"direction must be 'bullish' or 'bearish', got {direction!r}")

    # A level counts as "unbroken" only if no BOS/CHoCH in this direction
    # ever actually fired while it was the active swing level — otherwise
    # this just re-labels a level price already conquered (and later
    # retraced below/above again) as fresh resistance/support. `event_dir`
    # at bar i reflects a break of exactly the level in `idx_col` at that
    # bar (bos_choch.py updates the swing level, then checks/fires the
    # event, within the same iteration), so this is a direct read, not an
    # approximation.
    seen_idxs = []
    broken_idxs = set()
    last_idx = None
    for i in range(n):
        cur_idx = structure[idx_col].iloc[i]
        if pd.notna(cur_idx) and cur_idx != last_idx:
            seen_idxs.append(int(cur_idx))
            last_idx = cur_idx
        if pd.notna(cur_idx) and structure["event_direction"].iloc[i] == direction:
            broken_idxs.add(int(cur_idx))

    candidate_idxs = [idx for idx in seen_idxs if idx not in broken_idxs]

    if direction == "bullish":
        candidates = [price_col[idx] for idx in candidate_idxs if price_col[idx] > current_price]
        target = min(candidates) if candidates else None
    else:
        candidates = [price_col[idx] for idx in candidate_idxs if price_col[idx] < current_price]
        target = max(candidates) if candidates else None

    if target is None:
        return {"next_target": None, "next_target_pct": None, "note": "no older unbroken level found (possible blue-sky breakout)"}

    pct = (target / current_price - 1) * 100
    return {"next_target": float(target), "next_target_pct": float(pct), "note": None}


def _untaken_daily_swing_highs(daily_df: pd.DataFrame, daily_external_structure: pd.DataFrame) -> list[dict]:
    """Confirmed daily-external swing highs whose price has never been
    traded through (WICK-based: any later bar's HIGH, not just a close) —
    stricter than find_next_target's close-only definition above. Causal:
    only ever looks at daily_external_structure/daily_df up to the current
    last row, same as every other function in this project.
    """
    n = len(daily_df)
    highs = daily_df["high"].to_numpy()
    idx_col = daily_external_structure["swing_high_idx"]

    seen_idxs, last_idx = [], None
    for i in range(n):
        cur_idx = idx_col.iloc[i]
        if pd.notna(cur_idx) and cur_idx != last_idx:
            seen_idxs.append(int(cur_idx))
            last_idx = cur_idx

    levels = []
    for idx in seen_idxs:
        level_price = float(highs[idx])
        swept = idx + 1 < n and bool((highs[idx + 1:] > level_price).any())
        if not swept:
            levels.append({"price": level_price, "label": "prior swing high"})
    return levels


def _untaken_period_high(daily_df: pd.DataFrame, freq: str, label: str) -> dict | None:
    """Previous COMPLETED week's/month's high (previous_period_levels is
    causal — shift(1) per period), excluded if the CURRENT, still-forming
    period has already traded (wick) through it.
    """
    levels = previous_period_levels(daily_df, freq)
    level_price = levels["prev_period_high"].iloc[-1]
    if pd.isna(level_price):
        return None

    naive_idx = daily_df.index.tz_convert("UTC").tz_localize(None)
    period = pd.Series(naive_idx.to_period(freq), index=daily_df.index)
    this_period_bars = daily_df.loc[period == period.iloc[-1]]
    if bool((this_period_bars["high"] > level_price).any()):
        return None
    return {"price": float(level_price), "label": label}


def next_liquidity_targets(daily_df: pd.DataFrame, daily_external_structure: pd.DataFrame) -> dict:
    """The swing-trade dashboard's "where is price likely headed" display —
    the nearest one or two UNTAKEN liquidity levels above the current
    price: confirmed daily swing highs, the previous completed week's high,
    and the previous completed month's high (the daily/weekly-scale
    analogue of the forex Pine script's session-high/previous-day-high
    staged targets). Deliberately INFORMATIONAL ONLY — a 2026-07-27 3-agent
    investigation (research + backtest + independent stress-test, 35 NSE
    stocks/~10 years, each independently implemented) found that wiring a
    forced partial-exit into this nearest level actively destroys the
    swing-trade strategy's edge: on daily bars the nearest untaken
    liquidity sits a fraction of 1R from entry on average, so a forced
    target scratches exactly the small number of huge trend trades that
    supply most of the strategy's return (see entry_exit.py's config
    header for the full writeup, and the OFF-BY-DEFAULT optional
    partial_target there for anyone who wants that discipline anyway, at a
    distance far enough to be nearly inert).
    """
    current_price = float(daily_df["close"].iloc[-1])

    pool = _untaken_daily_swing_highs(daily_df, daily_external_structure)
    for freq, label in [("W", "previous week's high"), ("M", "previous month's high")]:
        level = _untaken_period_high(daily_df, freq, label)
        if level is not None:
            pool.append(level)

    # de-duplicate: a period's high often IS a confirmed swing high (the
    # same actual trading day), which would otherwise show the same price
    # twice under two different labels — keep the first (swing-high labels
    # are added before period labels above, so "prior swing high" wins ties)
    deduped: list[dict] = []
    seen_prices: list[float] = []
    for lvl in pool:
        if any(abs(lvl["price"] - p) < 1e-6 for p in seen_prices):
            continue
        deduped.append(lvl)
        seen_prices.append(lvl["price"])

    above = sorted((lvl for lvl in deduped if lvl["price"] > current_price), key=lambda lvl: lvl["price"])

    def _fmt(level: dict | None) -> dict:
        if level is None:
            return {"price": None, "label": None, "pct": None}
        return {"price": level["price"], "label": level["label"], "pct": (level["price"] / current_price - 1) * 100}

    t1 = _fmt(above[0] if len(above) > 0 else None)
    t2 = _fmt(above[1] if len(above) > 1 else None)
    return {"t1": t1, "t2": t2}
