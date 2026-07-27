import pandas as pd

from smc_validator.liquidity.daily_weekly import previous_period_levels


def test_previous_day_high_low_applies_to_every_bar_of_current_day():
    index = pd.date_range("2024-01-01 00:00", periods=48, freq="h", tz="UTC")
    df = pd.DataFrame({"high": [1.00] * 48, "low": [0.99] * 48}, index=index)
    df.loc[df.index[5], "high"] = 1.10  # day 1 high
    df.loc[df.index[10], "low"] = 0.90  # day 1 low

    levels = previous_period_levels(df, freq="D")

    # every bar in day 2 (indices 24-47) should see day 1's high/low
    assert (levels["prev_period_high"].iloc[24:48] == 1.10).all()
    assert (levels["prev_period_low"].iloc[24:48] == 0.90).all()
    # day 1 itself has no previous day yet
    assert levels["prev_period_high"].iloc[:24].isna().all()
