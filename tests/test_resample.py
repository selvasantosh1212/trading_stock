import pandas as pd

from smc_validator.data_ingestion.resample import align_htf_to_ltf, resample_ohlc


def test_resample_labels_bar_with_its_own_close_time():
    index = pd.date_range("2024-01-01 00:00", periods=8, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07],
            "high": [1.02, 1.03, 1.05, 1.04, 1.06, 1.07, 1.08, 1.09],
            "low": [0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06],
            "close": [1.01, 1.02, 1.03, 1.03, 1.05, 1.06, 1.07, 1.08],
        },
        index=index,
    )

    resampled = resample_ohlc(df, "4h")

    # a 4h bar starting 00:00 (bars 00,01,02,03) is labeled with its CLOSE
    # time, 04:00 — not its open time
    assert pd.Timestamp("2024-01-01 04:00", tz="UTC") in resampled.index
    assert pd.Timestamp("2024-01-01 00:00", tz="UTC") not in resampled.index

    first_bar = resampled.loc["2024-01-01 04:00"]
    assert first_bar["open"] == 1.00  # bar 00's open
    assert first_bar["high"] == 1.05  # max of bars 00-03
    assert first_bar["low"] == 0.99  # min of bars 00-03
    assert first_bar["close"] == 1.03  # bar 03's close


def test_align_htf_to_ltf_has_no_lookahead():
    htf_index = pd.to_datetime(["2024-01-01 04:00", "2024-01-01 08:00"], utc=True)
    htf_series = pd.Series(["bullish", "bearish"], index=htf_index)

    ltf_index = pd.to_datetime(
        ["2024-01-01 01:00", "2024-01-01 04:00", "2024-01-01 05:00", "2024-01-01 08:00", "2024-01-01 09:00"],
        utc=True,
    )

    aligned = align_htf_to_ltf(htf_series, ltf_index)

    # before the first HTF bar closes, nothing is known yet
    assert pd.isna(aligned.iloc[0])
    # exactly at / after the 04:00 close, that value becomes visible
    assert aligned.iloc[1] == "bullish"
    assert aligned.iloc[2] == "bullish"
    # the 08:00 HTF bar only becomes visible from 08:00 onward
    assert aligned.iloc[3] == "bearish"
    assert aligned.iloc[4] == "bearish"
