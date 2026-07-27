import pandas as pd


def previous_period_levels(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Previous COMPLETED period's high/low, forward-mapped onto every bar of
    the current (not-yet-complete) period — e.g. freq='D' gives each bar the
    previous calendar day's high/low; freq='W' the previous week's. Assumes
    df is sorted chronologically (true of everything loaded via
    data_ingestion in this package).
    """
    # storage convention is UTC everywhere (strategy_config.yaml timezone.storage),
    # so dropping tz info here before period-bucketing is safe and intentional
    naive_utc_index = df.index.tz_convert("UTC").tz_localize(None)
    period = pd.Series(naive_utc_index.to_period(freq), index=df.index)
    period_high = df.groupby(period)["high"].max().shift(1)
    period_low = df.groupby(period)["low"].min().shift(1)

    out = pd.DataFrame(index=df.index)
    out["prev_period_high"] = period.map(period_high)
    out["prev_period_low"] = period.map(period_low)
    return out
