from smc_validator.structure.bos_choch import compute_structure
from smc_validator.structure.swings import detect_fractal_pivots


def test_bos_fires_on_close_break_of_confirmed_swing_high(load_fixture):
    df = load_fixture("synthetic_uptrend_bos.csv")
    pivots = detect_fractal_pivots(df, left_bars=1, right_bars=1)
    structure = compute_structure(df, pivots, close_only=True)

    assert structure["event"].iloc[5] == "BOS"
    assert structure["event_direction"].iloc[5] == "bullish"
    assert structure["trend"].iloc[5] == "bullish"
    # no event should have fired before the actual break
    assert structure["event"].iloc[:5].isna().all()


def test_choch_fires_on_reversal_break_of_confirmed_swing_low(load_fixture):
    df = load_fixture("synthetic_choch_reversal.csv")
    pivots = detect_fractal_pivots(df, left_bars=1, right_bars=1)
    structure = compute_structure(df, pivots, close_only=True)

    assert structure["event"].iloc[5] == "BOS"
    assert structure["trend"].iloc[5] == "bullish"

    assert structure["event"].iloc[7] == "CHoCH"
    assert structure["event_direction"].iloc[7] == "bearish"
    assert structure["trend"].iloc[7] == "bearish"


def test_wick_beyond_level_does_not_count_when_close_only(load_fixture):
    df = load_fixture("synthetic_wick_no_break.csv")
    pivots = detect_fractal_pivots(df, left_bars=1, right_bars=1)

    close_only_structure = compute_structure(df, pivots, close_only=True)
    assert close_only_structure["event"].iloc[5] is None

    wick_structure = compute_structure(df, pivots, close_only=False)
    assert wick_structure["event"].iloc[5] == "BOS"
    assert wick_structure["event_direction"].iloc[5] == "bullish"


def test_event_does_not_refire_every_bar_beyond_a_stale_level(load_fixture):
    df = load_fixture("synthetic_choch_reversal.csv")
    pivots = detect_fractal_pivots(df, left_bars=1, right_bars=1)
    structure = compute_structure(df, pivots, close_only=True)

    events = structure["event"].dropna().tolist()
    assert events == ["BOS", "CHoCH"]
