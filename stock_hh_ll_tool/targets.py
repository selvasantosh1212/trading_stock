import pandas as pd


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
