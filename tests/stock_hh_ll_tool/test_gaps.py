import pandas as pd

from stock_hh_ll_tool.gaps import detect_overnight_gaps


def test_gap_up_detected_above_threshold():
    df = pd.DataFrame(
        {
            "open": [100.0, 103.0, 100.2],  # bar1: +3% gap up, bar2: +0.2% (below threshold)
            "high": [101.0, 104.0, 101.0],
            "low": [99.0, 102.0, 99.5],
            "close": [100.0, 103.5, 100.5],
        }
    )

    gaps = detect_overnight_gaps(df, min_gap_pct=0.5)

    assert gaps["gap_up"].iloc[1]
    assert not gaps["gap_down"].iloc[1]
    assert abs(gaps["gap_pct"].iloc[1] - 3.0) < 1e-9

    assert not gaps["gap_up"].iloc[2]  # 0.2% is below the 0.5% threshold


def test_gap_down_detected():
    df = pd.DataFrame(
        {"open": [100.0, 95.0], "high": [101.0, 96.0], "low": [99.0, 94.0], "close": [100.0, 95.5]}
    )

    gaps = detect_overnight_gaps(df, min_gap_pct=0.5)

    assert gaps["gap_down"].iloc[1]
    assert not gaps["gap_up"].iloc[1]
    assert abs(gaps["gap_pct"].iloc[1] - (-5.0)) < 1e-9


def test_first_bar_has_no_gap():
    df = pd.DataFrame({"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5]})
    gaps = detect_overnight_gaps(df)
    assert pd.isna(gaps["gap_pct"].iloc[0])
    assert not gaps["gap_up"].iloc[0]
    assert not gaps["gap_down"].iloc[0]
