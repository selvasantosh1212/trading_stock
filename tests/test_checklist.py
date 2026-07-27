import pandas as pd

from smc_validator.signal.checklist import assemble_signals
from smc_validator.signal.htf_bias import combine_bias
from smc_validator.signal.ltf_confirmation import detect_ltf_confirmation


def test_combine_bias_requires_all_timeframes_to_agree():
    idx = range(4)
    trend_4h = pd.Series(["bearish", "bearish", "bearish", "bullish"], index=idx)
    trend_1h = pd.Series(["bearish", "bullish", "bearish", "bullish"], index=idx)

    bias = combine_bias({"4H": trend_4h, "1H": trend_1h})

    assert bias.iloc[0] == "bearish"  # both agree bearish
    assert pd.isna(bias.iloc[1])  # conflict -> no bias
    assert bias.iloc[2] == "bearish"
    assert bias.iloc[3] == "bullish"  # both agree bullish


def test_ltf_confirmation_fires_once_on_flip_within_zone_and_timeout():
    idx = range(6)
    # LTF trend is bullish (opposite of bias) while price first enters the
    # zone, then flips to bearish (matching bias) at bar 3
    ltf_trend = pd.Series(["bullish", "bullish", "bullish", "bearish", "bearish", "bearish"], index=idx)
    htf_bias = pd.Series(["bearish"] * 6, index=idx)
    zone_active = pd.Series([False, True, True, True, True, False], index=idx)

    confirmation = detect_ltf_confirmation(ltf_trend, htf_bias, zone_active, timeout_bars=10)

    assert list(confirmation) == [False, False, False, True, False, False]


def test_ltf_confirmation_respects_timeout():
    idx = range(5)
    ltf_trend = pd.Series(["bullish", "bullish", "bullish", "bullish", "bearish"], index=idx)
    htf_bias = pd.Series(["bearish"] * 5, index=idx)
    zone_active = pd.Series([True, True, True, True, True], index=idx)

    # flip happens 4 bars after the zone started, but timeout is only 2 bars
    confirmation = detect_ltf_confirmation(ltf_trend, htf_bias, zone_active, timeout_bars=2)

    assert not confirmation.any()


def test_assemble_signals_end_to_end():
    idx = range(4)
    htf_bias = pd.Series(["bearish", "bearish", "bearish", "bearish"], index=idx)
    ltf_confirmation = pd.Series([False, False, True, False], index=idx)
    zone_direction = pd.Series(["bearish", "bearish", "bearish", "bullish"], index=idx)
    entry = pd.Series([1.00, 1.00, 0.98, 1.00], index=idx)
    stop = pd.Series([1.02, 1.02, 1.00, 1.02], index=idx)
    target1 = pd.Series([0.95, 0.95, 0.95, 0.95], index=idx)
    target2 = pd.Series([0.90, 0.90, 0.90, 0.90], index=idx)

    signals = assemble_signals(htf_bias, ltf_confirmation, zone_direction, entry, stop, target1, target2)

    assert len(signals) == 1
    assert signals.index[0] == 2
    row = signals.iloc[0]
    assert row["direction"] == "bearish"
    assert row["entry"] == 0.98
    assert row["stop"] == 1.00
    assert row["target1"] == 0.95
    assert row["target2"] == 0.90
