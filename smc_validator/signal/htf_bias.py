import pandas as pd


def combine_bias(trend_by_timeframe: dict[str, pd.Series]) -> pd.Series:
    """Combined directional bias across the bias timeframes (config:
    timeframe_roles.bias_timeframes, e.g. 4H + 1H). Per the user-confirmed
    rule, ALL given timeframes must agree — if they conflict, bias is None
    (no trade), matching the video's "15min bearish, 4H bearish" example.
    """
    trends = pd.DataFrame(trend_by_timeframe)
    all_bullish = (trends == "bullish").all(axis=1)
    all_bearish = (trends == "bearish").all(axis=1)

    bias = pd.Series(None, index=trends.index, dtype="object")
    bias[all_bullish] = "bullish"
    bias[all_bearish] = "bearish"
    return bias
