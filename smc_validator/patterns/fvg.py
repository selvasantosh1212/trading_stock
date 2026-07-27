import numpy as np
import pandas as pd


def detect_fvg(df: pd.DataFrame, min_gap_size: float = 0.0) -> pd.DataFrame:
    """3-candle imbalance: bullish FVG when low[i] > high[i-2] (a gap between
    candle i-2's high and candle i's low that candle i-1 never touched);
    bearish when high[i] < low[i-2]. `min_gap_size` filters noise gaps below
    an absolute threshold (caller derives this from ATR per strategy_config).
    """
    high, low = df["high"], df["low"]
    prior_high = high.shift(2)
    prior_low = low.shift(2)

    bullish_gap = low - prior_high
    bearish_gap = prior_low - high

    fvg_bullish = (bullish_gap > min_gap_size).fillna(False)
    fvg_bearish = (bearish_gap > min_gap_size).fillna(False)

    out = pd.DataFrame(index=df.index)
    out["fvg_bullish"] = fvg_bullish
    out["fvg_bearish"] = fvg_bearish
    out["fvg_top"] = pd.Series(np.nan, index=df.index, dtype="float64")
    out["fvg_bottom"] = pd.Series(np.nan, index=df.index, dtype="float64")
    out.loc[fvg_bullish, "fvg_top"] = low[fvg_bullish]
    out.loc[fvg_bullish, "fvg_bottom"] = prior_high[fvg_bullish]
    out.loc[fvg_bearish, "fvg_top"] = prior_low[fvg_bearish]
    out.loc[fvg_bearish, "fvg_bottom"] = high[fvg_bearish]
    return out


def mark_mitigation(df: pd.DataFrame, fvg_df: pd.DataFrame, rule: str = "close_crossing") -> pd.DataFrame:
    """Positional index (if any, relative to df, not a label) of the first
    bar where price rebalances each flagged FVG. `close_crossing` requires a
    candle CLOSE back through the gap boundary; `wick_crossing` accepts a
    wick — this materially changes results, so it's an explicit, exposed
    choice rather than a hardcoded default.
    """
    n = len(df)
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    fvg_bullish = fvg_df["fvg_bullish"].to_numpy()
    fvg_bearish = fvg_df["fvg_bearish"].to_numpy()
    fvg_top = fvg_df["fvg_top"].to_numpy()
    fvg_bottom = fvg_df["fvg_bottom"].to_numpy()

    mitigated_at: list[int | None] = [None] * n

    for i in range(n):
        if fvg_bullish[i]:
            bottom = fvg_bottom[i]
            ref = close if rule == "close_crossing" else low
            for j in range(i + 1, n):
                if ref[j] <= bottom:
                    mitigated_at[i] = j
                    break
        elif fvg_bearish[i]:
            top = fvg_top[i]
            ref = close if rule == "close_crossing" else high
            for j in range(i + 1, n):
                if ref[j] >= top:
                    mitigated_at[i] = j
                    break

    return pd.DataFrame({"mitigated_at": mitigated_at}, index=df.index)
