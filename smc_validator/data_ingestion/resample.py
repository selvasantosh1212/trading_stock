import pandas as pd


def resample_ohlc(df: pd.DataFrame, target_freq: str) -> pd.DataFrame:
    """Resample a base-timeframe OHLC DataFrame to a higher timeframe.

    Base bars are timestamped by their OPEN time (standard OHLC convention —
    a bar labeled "09:00" covers [09:00, 10:00)). `closed="left"` groups each
    higher-timeframe bin as [start, end) so it collects exactly the base
    bars whose own open falls in that range; `label="right"` then stamps the
    resulting bar with `end` (its actual close time). Together this is what
    keeps multi-timeframe alignment lookahead-safe — a resampled bar's
    timestamp is never earlier than the moment all of its inputs existed.
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    resampled = df.resample(target_freq, label="right", closed="left").agg(agg)
    return resampled.dropna(how="any")


def align_htf_to_ltf(htf_series: pd.Series, ltf_index: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill an HTF series (indexed by its own bar CLOSE times) onto
    an LTF index. `merge_asof(direction="backward")` guarantees each LTF bar
    only ever sees HTF values already closed as of its own timestamp — no
    lookahead into a still-forming HTF bar.
    """
    htf_df = htf_series.rename("value").reset_index()
    htf_df.columns = ["timestamp", "value"]
    ltf_df = pd.DataFrame({"timestamp": ltf_index})

    merged = pd.merge_asof(
        ltf_df.sort_values("timestamp"),
        htf_df.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    return pd.Series(merged["value"].to_numpy(), index=ltf_index)
