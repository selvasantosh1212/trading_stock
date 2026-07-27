import pandas as pd

from stock_hh_ll_tool.config import load_config
from stock_hh_ll_tool.entry_exit import FLAT_POSITION, evaluate_position_transition

CFG = {
    **load_config(),
    "entry_exit_strategy": {
        "enabled": True,
        "htf_structure": {"left_bars": 1, "right_bars": 1},
        "ltf_structure": {"left_bars": 1, "right_bars": 1},
        "stop_atr_buffer_fraction": 0.0,  # no buffer, keeps expected numbers exact in tests
        "risk_per_trade_pct": 1.0,
    },
}


def _bearish_then_bullish_choch_df(periods: int) -> pd.DataFrame:
    # mirror image (2.0 - value, high/low swapped) of tests/fixtures/synthetic_choch_reversal.csv,
    # which is proven (test_bos_choch.py) to produce BOS at bar5 then CHoCH
    # at bar7 in the BULLISH direction — mirroring flips both to BEARISH
    # BOS at bar5 then BULLISH CHoCH at bar7, which is exactly the
    # retracement-then-reversal shape this strategy enters on.
    original = pd.DataFrame(
        {
            "open": [1.00, 0.99, 0.98, 1.01, 1.04, 1.02, 1.07, 1.06],
            "high": [1.00, 1.00, 1.02, 1.05, 1.03, 1.08, 1.075, 1.065],
            "low": [0.98, 0.97, 0.99, 1.00, 1.01, 1.015, 1.05, 0.95],
            "close": [0.99, 0.98, 1.01, 1.04, 1.02, 1.07, 1.06, 0.96],
        }
    )
    mirrored = pd.DataFrame(
        {
            "open": 2.0 - original["open"],
            "high": 2.0 - original["low"],
            "low": 2.0 - original["high"],
            "close": 2.0 - original["close"],
        }
    )
    index = pd.date_range("2026-03-02", periods=periods, freq="D", tz="UTC")
    return mirrored.iloc[:periods].set_index(index)


def _bullish_weekly_df() -> pd.DataFrame:
    # same values as tests/fixtures/synthetic_uptrend_bos.csv (proven in
    # test_bos_choch.py to produce a bullish BOS at bar5 with left=1/right=1
    # pivots) — needs at least left+right+1 bars per bar to detect any
    # pivot at all, so this can't be shrunk further
    index = pd.date_range("2025-12-07", periods=6, freq="W", tz="UTC")
    return pd.DataFrame(
        {
            "open": [1.00, 0.99, 0.98, 1.01, 1.04, 1.02],
            "high": [1.00, 1.00, 1.02, 1.05, 1.03, 1.08],
            "low": [0.98, 0.97, 0.99, 1.00, 1.01, 1.015],
            "close": [0.99, 0.98, 1.01, 1.04, 1.02, 1.07],
        },
        index=index,
    )


def test_flat_stays_flat_before_the_choch_bar():
    weekly = _bullish_weekly_df()
    daily = _bearish_then_bullish_choch_df(7)  # truncated before the actual CHoCH bar (idx7)

    result = evaluate_position_transition(weekly, daily, CFG, dict(FLAT_POSITION))

    assert result["new_position"]["state"] == "FLAT"
    assert result["event"] is None


def test_flat_signals_entry_on_bullish_choch_bar():
    weekly = _bullish_weekly_df()
    daily = _bearish_then_bullish_choch_df(8)  # includes the CHoCH bar (idx7)

    result = evaluate_position_transition(weekly, daily, CFG, dict(FLAT_POSITION))

    assert result["new_position"]["state"] == "PENDING_ENTRY"
    assert result["event"]["type"] == "signal_entry"
    assert result["new_position"]["stop_price"] is not None


def test_pending_entry_fills_at_todays_open_not_signal_close():
    weekly = _bullish_weekly_df()
    daily = _bearish_then_bullish_choch_df(8)
    # simulate: yesterday's run set PENDING_ENTRY with some stop; today we
    # just need ANY new daily bar appended to represent "today"
    extra_day = pd.DataFrame(
        {"open": [1.10], "high": [1.12], "low": [1.08], "close": [1.11]},
        index=pd.date_range(daily.index[-1] + pd.Timedelta(days=1), periods=1, freq="D", tz="UTC"),
    )
    daily_today = pd.concat([daily, extra_day])
    pending = {"state": "PENDING_ENTRY", "stop_price": 0.90, "entry_price": None, "entry_date": None}

    result = evaluate_position_transition(weekly, daily_today, CFG, pending)

    assert result["new_position"]["state"] == "LONG"
    assert result["event"]["type"] == "entered"
    assert result["event"]["entry_price"] == 1.10  # today's OPEN, not close
    assert result["new_position"]["stop_price"] == 0.90  # carried over unchanged


def test_long_exits_on_stop_breach():
    weekly = _bullish_weekly_df()
    daily = _bearish_then_bullish_choch_df(8)
    position = {"state": "LONG", "entry_price": 1.00, "entry_date": "2026-01-01", "stop_price": 0.97}
    # last bar's low in the fixture (idx7) is 2.0-1.065=0.935, below the 0.97 stop
    assert daily["low"].iloc[-1] < 0.97

    result = evaluate_position_transition(weekly, daily, CFG, position)

    assert result["new_position"]["state"] == "FLAT"
    assert result["event"]["type"] == "exit_stop"
    assert result["event"]["exit_price"] == 0.97


def test_long_exits_on_htf_trend_reversal():
    # same values as tests/fixtures/synthetic_choch_reversal.csv (proven in
    # test_bos_choch.py: bullish BOS at bar5, then bearish CHoCH at bar7) —
    # weekly trend ends up "bearish" as of the last bar
    index = pd.date_range("2025-11-23", periods=8, freq="W", tz="UTC")
    weekly = pd.DataFrame(
        {
            "open": [1.00, 0.99, 0.98, 1.01, 1.04, 1.02, 1.07, 1.06],
            "high": [1.00, 1.00, 1.02, 1.05, 1.03, 1.08, 1.075, 1.065],
            "low": [0.98, 0.97, 0.99, 1.00, 1.01, 1.015, 1.05, 0.95],
            "close": [0.99, 0.98, 1.01, 1.04, 1.02, 1.07, 1.06, 0.96],
        },
        index=index,
    )
    daily = _bearish_then_bullish_choch_df(8)
    position = {"state": "LONG", "entry_price": 1.00, "entry_date": "2026-01-01", "stop_price": 0.50}  # stop far away, won't trigger

    result = evaluate_position_transition(weekly, daily, CFG, position)

    assert result["new_position"]["state"] == "FLAT"
    assert result["event"]["type"] == "exit_trend"


def test_long_holds_when_no_exit_condition_met():
    weekly = _bullish_weekly_df()
    daily = _bearish_then_bullish_choch_df(6)  # before any bearish HTF flip, stop far away
    position = {"state": "LONG", "entry_price": 1.00, "entry_date": "2026-01-01", "stop_price": 0.10}

    result = evaluate_position_transition(weekly, daily, CFG, position)

    assert result["new_position"]["state"] == "LONG"
    assert result["event"] is None
