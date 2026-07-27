from zoneinfo import ZoneInfo

import pandas as pd


def _session_mask(index_utc: pd.DatetimeIndex, tz_name: str, start: str, end: str) -> pd.Series:
    local = index_utc.tz_convert(ZoneInfo(tz_name))
    start_h, start_m = (int(x) for x in start.split(":"))
    end_h, end_m = (int(x) for x in end.split(":"))
    local_minutes = local.hour * 60 + local.minute
    start_minutes = start_h * 60 + start_m
    end_minutes = end_h * 60 + end_m
    return pd.Series((local_minutes >= start_minutes) & (local_minutes < end_minutes), index=index_utc)


def compute_session_levels(df: pd.DataFrame, tz_name: str, start: str, end: str) -> pd.DataFrame:
    """Per-bar running high/low of the current session instance (using the
    session's own IANA timezone, not a fixed UTC offset — offsets silently
    break across DST). Levels freeze at session close and stay forward-filled
    (via ffill) until the next instance starts, so downstream code always has
    "the last completed/in-progress session's levels" available at any bar.
    """
    mask = _session_mask(df.index, tz_name, start, end)
    session_start = mask & ~mask.shift(1, fill_value=False)
    instance_id = session_start.cumsum().where(mask)

    high = df["high"].where(mask)
    low = df["low"].where(mask)

    session_high = high.groupby(instance_id).cummax()
    session_low = low.groupby(instance_id).cummin()

    out = pd.DataFrame(index=df.index)
    out["in_session"] = mask
    out["session_instance_id"] = instance_id
    out["session_high"] = session_high.ffill()
    out["session_low"] = session_low.ffill()
    return out
