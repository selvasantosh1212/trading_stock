import pandas as pd

from smc_validator.liquidity.sessions import compute_session_levels


def _make_hourly_df(start: str, hours: int, highs: list[float], lows: list[float]) -> pd.DataFrame:
    index = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    return pd.DataFrame({"high": highs, "low": lows}, index=index)


def test_session_levels_track_running_high_low_within_window():
    # 09:00-17:00 UTC session, day 1: 06:00 through 20:00 (15 hourly bars)
    highs = [1.00, 1.00, 1.00, 1.05, 1.03, 1.08, 1.02, 1.01, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
    lows = [0.99, 0.99, 0.99, 0.98, 0.99, 0.99, 0.95, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
    df = _make_hourly_df("2024-01-01 06:00", 15, highs, lows)

    levels = compute_session_levels(df, tz_name="UTC", start="09:00", end="17:00")

    # bars before 09:00 are outside the session
    assert not levels["in_session"].iloc[0]
    # 09:00 is bar index 3 (06,07,08,09...)
    assert levels["in_session"].iloc[3]
    assert not levels["in_session"].iloc[11]  # 17:00 bar itself is exclusive end

    # running high/low within the session should reflect the max/min seen so far
    assert levels["session_high"].iloc[5] == 1.08  # 11:00 bar itself has the 1.08 high
    assert levels["session_low"].iloc[6] == 0.95  # by 12:00, the 0.95 low is in


def test_session_levels_freeze_and_forward_fill_after_session_ends():
    highs = [1.00, 1.00, 1.00, 1.05, 1.03, 1.08, 1.02, 1.01, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
    lows = [0.99, 0.99, 0.99, 0.98, 0.99, 0.99, 0.95, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]
    df = _make_hourly_df("2024-01-01 06:00", 15, highs, lows)

    levels = compute_session_levels(df, tz_name="UTC", start="09:00", end="17:00")

    final_high, final_low = levels["session_high"].iloc[10], levels["session_low"].iloc[10]
    # bars after the session ends (17:00 onward) keep the frozen final level
    assert levels["session_high"].iloc[12] == final_high
    assert levels["session_low"].iloc[12] == final_low


def test_new_session_instance_starts_fresh_the_next_day():
    # two days of a 09:00-17:00 session; day 2 has a distinctly different range
    hours = 48
    highs = [1.00] * hours
    lows = [0.99] * hours
    df = _make_hourly_df("2024-01-01 00:00", hours, highs, lows)
    df.loc[df.index[9], "high"] = 1.10  # day 1, 09:00 session, spike
    df.loc[df.index[33], "high"] = 1.20  # day 2, 09:00 session, different spike

    levels = compute_session_levels(df, tz_name="UTC", start="09:00", end="17:00")

    assert levels["session_high"].iloc[16] == 1.10  # end of day 1 session
    assert levels["session_high"].iloc[40] == 1.20  # end of day 2 session, independent of day 1
