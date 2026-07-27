from smc_validator.structure.swings import detect_fractal_pivots


def test_pivot_low_detected_and_confirmed_with_lag(load_fixture):
    df = load_fixture("synthetic_uptrend_bos.csv")
    pivots = detect_fractal_pivots(df, left_bars=1, right_bars=1)

    assert pivots["pivot_low"].iloc[1]
    assert pivots["confirmed_at"].iloc[1] == 2
    assert not pivots["pivot_low"].iloc[2]


def test_pivot_high_detected_and_confirmed_with_lag(load_fixture):
    df = load_fixture("synthetic_uptrend_bos.csv")
    pivots = detect_fractal_pivots(df, left_bars=1, right_bars=1)

    assert pivots["pivot_high"].iloc[3]
    assert pivots["confirmed_at"].iloc[3] == 4
    assert not pivots["pivot_high"].iloc[4]


def test_small_lookback_finds_more_pivots_than_large_lookback(load_fixture):
    df = load_fixture("choppy_internal_leg.csv")
    small = detect_fractal_pivots(df, left_bars=1, right_bars=1)
    large = detect_fractal_pivots(df, left_bars=3, right_bars=3)

    small_count = int((small["pivot_high"] | small["pivot_low"]).sum())
    large_count = int((large["pivot_high"] | large["pivot_low"]).sum())
    assert small_count > large_count


def test_choppy_dip_and_wiggle_are_small_lookback_pivots_only(load_fixture):
    df = load_fixture("choppy_internal_leg.csv")
    small = detect_fractal_pivots(df, left_bars=1, right_bars=1)
    large = detect_fractal_pivots(df, left_bars=3, right_bars=3)

    assert small["pivot_low"].iloc[4]
    assert not large["pivot_low"].iloc[4]

    assert small["pivot_high"].iloc[7]
    assert not large["pivot_high"].iloc[7]
