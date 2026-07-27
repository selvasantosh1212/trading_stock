from smc_validator.patterns.fvg import detect_fvg, mark_mitigation


def test_bearish_fvg_detected(load_fixture):
    df = load_fixture("bearish_ob_with_fvg.csv")
    fvg = detect_fvg(df)

    assert fvg["fvg_bearish"].iloc[6]
    assert not fvg["fvg_bullish"].iloc[6]
    assert fvg["fvg_top"].iloc[6] == df["low"].iloc[4]
    assert fvg["fvg_bottom"].iloc[6] == df["high"].iloc[6]


def test_min_gap_size_filters_small_gaps(load_fixture):
    df = load_fixture("bearish_ob_with_fvg.csv")
    fvg = detect_fvg(df, min_gap_size=1.0)  # threshold far larger than any real gap

    assert not fvg["fvg_bearish"].any()
    assert not fvg["fvg_bullish"].any()


def test_mitigation_close_crossing_finds_rebalance_bar(load_fixture):
    df = load_fixture("bearish_ob_with_fvg.csv")
    fvg = detect_fvg(df)
    mitigation = mark_mitigation(df, fvg, rule="close_crossing")

    # bar 6's bearish fvg top (far boundary) = 0.9900 (low[4]); close later
    # fully rallies back above it at bar 11 (close=0.9910) — "mitigated"
    # here means the full gap got filled, not just a first touch
    assert mitigation["mitigated_at"].iloc[6] == 11
