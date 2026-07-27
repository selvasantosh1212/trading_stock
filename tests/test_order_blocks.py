import pandas as pd

from smc_validator.patterns.fvg import detect_fvg
from smc_validator.patterns.order_blocks import detect_order_blocks


def _constant_atr(df, value=0.01):
    return pd.Series(value, index=df.index)


def test_last_balanced_candle_before_expansion_flagged_bearish_ob(load_fixture):
    df = load_fixture("bearish_ob_with_fvg.csv")
    atr = _constant_atr(df)
    fvg = detect_fvg(df)

    obs = detect_order_blocks(df, atr, lookahead=5, fvg_flags=fvg)

    assert obs["bearish_ob"].iloc[3]
    assert not obs["bullish_ob"].iloc[3]


def test_earlier_balanced_candles_not_flagged_only_the_last_one_is(load_fixture):
    df = load_fixture("bearish_ob_with_fvg.csv")
    atr = _constant_atr(df)
    fvg = detect_fvg(df)

    obs = detect_order_blocks(df, atr, lookahead=5, fvg_flags=fvg)

    # bars 0-2 are also balanced and precede enough eventual expansion, but
    # their immediate next candle is still balanced too — only bar 3 (whose
    # next candle starts the real expansion) should qualify
    assert not obs["bearish_ob"].iloc[0]
    assert not obs["bearish_ob"].iloc[1]
    assert not obs["bearish_ob"].iloc[2]


def test_require_fvg_filters_out_candidates_without_one(load_fixture):
    df = load_fixture("bearish_ob_with_fvg.csv")
    atr = _constant_atr(df)
    # a min_gap_size far larger than the real gap suppresses the FVG entirely
    fvg_none = detect_fvg(df, min_gap_size=1.0)

    obs = detect_order_blocks(df, atr, lookahead=5, fvg_flags=fvg_none)

    assert not obs["bearish_ob"].any()
