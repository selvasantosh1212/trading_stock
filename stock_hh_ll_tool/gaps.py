import pandas as pd


def detect_overnight_gaps(daily_df: pd.DataFrame, min_gap_pct: float = 0.5) -> pd.DataFrame:
    """Overnight gap = today's open vs yesterday's close, as a percentage.
    A gap up alongside a confirmed Higher High is a stronger signal than
    either alone (a common swing-trading pattern: breakaway gap on the same
    day structure confirms). `min_gap_pct` filters noise gaps.
    """
    prev_close = daily_df["close"].shift(1)
    gap_pct = (daily_df["open"] - prev_close) / prev_close * 100

    out = pd.DataFrame(index=daily_df.index)
    out["prev_close"] = prev_close
    out["gap_pct"] = gap_pct
    out["gap_up"] = gap_pct > min_gap_pct
    out["gap_down"] = gap_pct < -min_gap_pct
    return out
