import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return true_range(df).rolling(window).mean()


def _leading_max(series: pd.Series, window: int) -> pd.Series:
    return series[::-1].rolling(window, min_periods=1).max()[::-1]


def _leading_min(series: pd.Series, window: int) -> pd.Series:
    return series[::-1].rolling(window, min_periods=1).min()[::-1]


def detect_order_blocks(
    df: pd.DataFrame,
    atr_series: pd.Series,
    balance_body_atr_fraction: float = 0.5,
    expansion_atr_multiple: float = 2.0,
    lookahead: int = 5,
    fvg_flags: pd.DataFrame | None = None,
    structure_event: pd.Series | None = None,
    structure_event_direction: pd.Series | None = None,
) -> pd.DataFrame:
    """An order block is the LAST balanced candle (small body relative to
    ATR, and whose immediate next candle is no longer balanced — i.e. it is
    the final ranging bar before the move starts, not just any ranging bar)
    immediately preceding an aggressive expansion (net displacement over the
    next `lookahead` bars beyond `expansion_atr_multiple` * ATR) in the
    opposite direction. If `fvg_flags` (from `patterns.fvg.detect_fvg`) is
    given, only keep candidates with a same-direction FVG within the
    lookahead window — the video's explicit "a good order block should have
    an associated FVG" confluence rule.

    An ATR-multiple price move alone is a very loose filter on real
    intraday data (empirically ~1 candidate every ~12 bars on 15-minute
    EURUSD — far more often than genuine "smart money" zones should form).
    Passing `structure_event`/`structure_event_direction` (the `event`/
    `event_direction` columns from `structure.bos_choch.compute_structure`
    on this same timeframe) additionally requires the expansion to actually
    cause a confirmed BOS/CHoCH in the matching direction within the
    lookahead window — tying order blocks to genuine structural
    significance, not just any sufficiently large candle range.
    """
    body = (df["close"] - df["open"]).abs()
    is_balanced = body <= balance_body_atr_fraction * atr_series
    # NOTE: shift(-1).fillna(False) on a bool Series upcasts to object dtype;
    # applying `~` to that does Python bitwise inversion (~True == -2, both
    # truthy) instead of logical negation — must force back to bool first.
    next_is_balanced = is_balanced.shift(-1).fillna(False).astype(bool)
    is_last_balanced = is_balanced & ~next_is_balanced

    future_max_high = _leading_max(df["high"], lookahead)
    future_min_low = _leading_min(df["low"], lookahead)

    bullish_expansion = (future_max_high - df["close"]) >= expansion_atr_multiple * atr_series
    bearish_expansion = (df["close"] - future_min_low) >= expansion_atr_multiple * atr_series

    bullish_ob = (is_last_balanced & bullish_expansion).fillna(False)
    bearish_ob = (is_last_balanced & bearish_expansion).fillna(False)

    if fvg_flags is not None:
        fvg_bullish_ahead = _leading_max(fvg_flags["fvg_bullish"].astype(int), lookahead) > 0
        fvg_bearish_ahead = _leading_max(fvg_flags["fvg_bearish"].astype(int), lookahead) > 0
        bullish_ob = bullish_ob & fvg_bullish_ahead
        bearish_ob = bearish_ob & fvg_bearish_ahead

    if structure_event is not None:
        is_bullish_event = (structure_event.notna()) & (structure_event_direction == "bullish")
        is_bearish_event = (structure_event.notna()) & (structure_event_direction == "bearish")
        bullish_event_ahead = _leading_max(is_bullish_event.astype(int), lookahead) > 0
        bearish_event_ahead = _leading_max(is_bearish_event.astype(int), lookahead) > 0
        bullish_ob = bullish_ob & bullish_event_ahead
        bearish_ob = bearish_ob & bearish_event_ahead

    out = pd.DataFrame(index=df.index)
    out["bullish_ob"] = bullish_ob
    out["bearish_ob"] = bearish_ob
    return out
