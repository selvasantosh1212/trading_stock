import pandas as pd

from stock_hh_ll_tool.config import load_config
from stock_hh_ll_tool.screener import evaluate_daily_signal, evaluate_intraday_bias

CFG = load_config()


def _bullish_bos_df() -> pd.DataFrame:
    # same scenario proven in smc_validator's synthetic_uptrend_bos.csv,
    # with left=1/right=1 pivots — override the config's left/right to match
    index = pd.date_range("2026-01-01", periods=6, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": [1.00, 0.99, 0.98, 1.01, 1.04, 1.02],
            "high": [1.00, 1.00, 1.02, 1.05, 1.03, 1.08],
            "low": [0.98, 0.97, 0.99, 1.00, 1.01, 1.015],
            "close": [0.99, 0.98, 1.01, 1.04, 1.02, 1.07],
        },
        index=index,
    )


def test_evaluate_daily_signal_flags_hh_broken_today():
    cfg = {**CFG, "daily_structure": {"external": {"left_bars": 1, "right_bars": 1}, "close_only_break": True}}
    df = _bullish_bos_df()

    result = evaluate_daily_signal(df, cfg)

    assert result["hh_broken_today"] is True
    assert result["structure_event"] == "BOS"
    assert result["daily_trend"] == "bullish"
    assert result["close"] == 1.07


def test_evaluate_daily_signal_no_signal_when_nothing_broke():
    cfg = {**CFG, "daily_structure": {"external": {"left_bars": 1, "right_bars": 1}, "close_only_break": True}}
    df = _bullish_bos_df().iloc[:4]  # truncate before the actual break bar

    result = evaluate_daily_signal(df, cfg)

    assert result["hh_broken_today"] is False


def test_evaluate_daily_signal_detects_gap_up_alongside_break():
    cfg = {
        **CFG,
        "daily_structure": {"external": {"left_bars": 1, "right_bars": 1}, "close_only_break": True},
        "gap": {"min_gap_pct": 0.5},
    }
    df = _bullish_bos_df().copy()
    df.loc[df.index[-1], "open"] = df["close"].iloc[-2] * 1.03  # +3% gap up on the break day

    result = evaluate_daily_signal(df, cfg)

    assert result["hh_broken_today"] is True
    assert result["gapped_up_today"] is True
    assert result["gap_pct"] > 2.9


def test_evaluate_intraday_bias_requires_all_bias_timeframes_to_match_direction():
    cfg = {**CFG, "intraday_confirmation": {**CFG["intraday_confirmation"], "bias_timeframes": ["4h", "1h"]}}

    agrees = evaluate_intraday_bias({"4h": "bullish", "1h": "bullish"}, cfg, expected_direction="bullish")
    assert agrees["bias_timeframes_agree"] is True

    conflict = evaluate_intraday_bias({"4h": "bullish", "1h": "bearish"}, cfg, expected_direction="bullish")
    assert conflict["bias_timeframes_agree"] is False

    bearish_case = evaluate_intraday_bias({"4h": "bearish", "1h": "bearish"}, cfg, expected_direction="bearish")
    assert bearish_case["bias_timeframes_agree"] is True
