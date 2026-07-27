import pandas as pd


def detect_fractal_pivots(df: pd.DataFrame, left_bars: int, right_bars: int) -> pd.DataFrame:
    """Fractal swing pivots: bar i is a pivot high/low if it is the unique
    extreme within [i-left_bars, i+right_bars]. A pivot is only knowable
    `right_bars` bars after it forms (`confirmed_at`) — callers must never use
    a pivot before its confirmation bar, or results become lookahead-biased.

    Same routine is reused for both internal (small lookback) and external
    (large lookback) structure — only the left/right params differ.
    """
    high = df["high"]
    low = df["low"]
    n = len(df)

    pivot_high = pd.Series(False, index=df.index)
    pivot_low = pd.Series(False, index=df.index)
    confirmed_at = pd.Series(pd.NA, index=df.index, dtype="Int64")

    for i in range(left_bars, n - right_bars):
        window_high = high.iloc[i - left_bars : i + right_bars + 1]
        window_low = low.iloc[i - left_bars : i + right_bars + 1]

        if high.iloc[i] == window_high.max() and (window_high == high.iloc[i]).sum() == 1:
            pivot_high.iloc[i] = True
            confirmed_at.iloc[i] = i + right_bars

        if low.iloc[i] == window_low.min() and (window_low == low.iloc[i]).sum() == 1:
            pivot_low.iloc[i] = True
            confirmed_at.iloc[i] = i + right_bars

    return pd.DataFrame(
        {"pivot_high": pivot_high, "pivot_low": pivot_low, "confirmed_at": confirmed_at},
        index=df.index,
    )
