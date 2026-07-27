import pandas as pd


def detect_sweep(df: pd.DataFrame, level: pd.Series, direction: str) -> pd.Series:
    """A sweep = this bar's price takes out a given reference level.
    `level` must already be whatever prior/frozen level the caller wants
    tested (e.g. previous session's high, previous day's low) — this stays a
    pure "did price take it out" check with no lookahead of its own.
    """
    if direction == "high":
        return df["high"] > level
    if direction == "low":
        return df["low"] < level
    raise ValueError(f"direction must be 'high' or 'low', got {direction!r}")


def first_sweep_then_reverse(
    df: pd.DataFrame, first_level: pd.Series, first_direction: str, second_level: pd.Series, second_direction: str
) -> pd.Series:
    """Flags bars where the OPPOSITE side gets swept after the first side was
    already swept earlier in the same lookback window — the "sweep one side,
    reverse, take the other" pattern described for session liquidity. Both
    swept flags must have already occurred by this bar, with the first
    strictly before the second.
    """
    first_swept = detect_sweep(df, first_level, first_direction)
    second_swept = detect_sweep(df, second_level, second_direction)
    first_swept_so_far = first_swept.cummax()
    return second_swept & first_swept_so_far.shift(1, fill_value=False)
