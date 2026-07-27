from typing import Optional

import pandas as pd


def compute_structure(df: pd.DataFrame, pivots: pd.DataFrame, close_only: bool = True) -> pd.DataFrame:
    """Sequential close-based BOS/CHoCH state machine.

    `swing_high`/`swing_low` always track the most recently CONFIRMED pivot of
    each type (never used before its `confirmed_at` bar — keeps this
    lookahead-safe). An event fires the bar a candle's reference price
    (close if close_only, else high/low) first crosses the current swing
    level; per the source rule, a wick beyond the level does NOT count when
    close_only=True. Each level only fires once per crossing (`*_broken`
    flags) until a newer pivot confirms and resets it — otherwise every bar
    price remains beyond a stale level would re-fire the same event.
    """
    n = len(df)
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    pivot_high = pivots["pivot_high"].to_numpy()
    pivot_low = pivots["pivot_low"].to_numpy()
    confirmed_at = pivots["confirmed_at"]

    confirms_high: dict[int, list[int]] = {}
    confirms_low: dict[int, list[int]] = {}
    for i in range(n):
        if pivot_high[i]:
            confirms_high.setdefault(int(confirmed_at.iloc[i]), []).append(i)
        if pivot_low[i]:
            confirms_low.setdefault(int(confirmed_at.iloc[i]), []).append(i)

    swing_high: Optional[float] = None
    swing_high_idx: Optional[int] = None
    swing_low: Optional[float] = None
    swing_low_idx: Optional[int] = None
    trend: Optional[str] = None
    high_broken = False
    low_broken = False

    out_trend: list[Optional[str]] = [None] * n
    out_event: list[Optional[str]] = [None] * n
    out_event_dir: list[Optional[str]] = [None] * n
    out_swing_high: list[Optional[float]] = [None] * n
    out_swing_high_idx: list[Optional[int]] = [None] * n
    out_swing_low: list[Optional[float]] = [None] * n
    out_swing_low_idx: list[Optional[int]] = [None] * n

    for i in range(n):
        for origin in confirms_high.get(i, []):
            swing_high, swing_high_idx = high[origin], origin
            high_broken = False
        for origin in confirms_low.get(i, []):
            swing_low, swing_low_idx = low[origin], origin
            low_broken = False

        ref_high = close[i] if close_only else high[i]
        ref_low = close[i] if close_only else low[i]

        if swing_high is not None and not high_broken and ref_high > swing_high:
            out_event[i] = "CHoCH" if trend == "bearish" else "BOS"
            out_event_dir[i] = "bullish"
            trend = "bullish"
            high_broken = True
        elif swing_low is not None and not low_broken and ref_low < swing_low:
            out_event[i] = "CHoCH" if trend == "bullish" else "BOS"
            out_event_dir[i] = "bearish"
            trend = "bearish"
            low_broken = True

        out_trend[i] = trend
        out_swing_high[i] = swing_high
        out_swing_high_idx[i] = swing_high_idx
        out_swing_low[i] = swing_low
        out_swing_low_idx[i] = swing_low_idx

    return pd.DataFrame(
        {
            "trend": out_trend,
            "event": out_event,
            "event_direction": out_event_dir,
            "swing_high": out_swing_high,
            "swing_high_idx": out_swing_high_idx,
            "swing_low": out_swing_low,
            "swing_low_idx": out_swing_low_idx,
        },
        index=df.index,
    )
