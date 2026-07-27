import pandas as pd


def detect_ltf_confirmation(
    ltf_trend: pd.Series, htf_bias: pd.Series, zone_active: pd.Series, timeout_bars: int
) -> pd.Series:
    """Confirmation fires the bar the entry-timeframe trend (config:
    timeframe_roles.entry_timeframes, e.g. 15min/5min) FLIPS to match the
    HTF bias, provided that happens while price is still inside the zone of
    interest (an OB+FVG confluence zone — see patterns.order_blocks /
    patterns.fvg) and within `timeout_bars` bars of the zone becoming
    active. A stale zone that's gone unconfirmed too long should not fire —
    per the source rule, location alone is never sufficient without this
    confirmation step.
    """
    matches_bias = (ltf_trend == htf_bias) & htf_bias.notna()
    just_flipped = matches_bias & ~matches_bias.shift(1).fillna(False).astype(bool)

    zone_start = zone_active & ~zone_active.shift(1).fillna(False).astype(bool)
    bars_since_zone_start = zone_active.groupby(zone_start.cumsum()).cumcount()
    within_timeout = bars_since_zone_start <= timeout_bars

    return just_flipped & zone_active & within_timeout
