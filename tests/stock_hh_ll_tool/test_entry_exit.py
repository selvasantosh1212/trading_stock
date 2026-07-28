import pandas as pd

from stock_hh_ll_tool.config import load_config
from stock_hh_ll_tool.entry_exit import FLAT_POSITION, evaluate_position_transition

# Shared CFG for the ARM/entry tests: daily_external uses a WIDE (3,3)
# window so it stays anchored through a shallow internal-only pullback;
# daily_internal uses a NARROW (1,1) window so it actually reacts to that
# pullback. weekly stays 1/1 to match the small `_bullish_weekly_df` fixture.
CFG = {
    **load_config(),
    "entry_exit_strategy": {
        "enabled": True,
        "weekly_structure": {"left_bars": 1, "right_bars": 1},
        "daily_external_structure": {"left_bars": 3, "right_bars": 3},
        "daily_internal_structure": {"left_bars": 1, "right_bars": 1},
        "stop_atr_buffer_fraction": 0.0,  # no buffer, keeps expected numbers exact in tests
        "risk_per_trade_pct": 1.0,
        "partial_target": {"enabled": False, "min_r": 3.0, "fraction": 0.25},
    },
}

# A separate CFG for exit/fill tests that reuse the small 8-bar mirrored
# fixture below — that fixture is too short for a (3,3) external window to
# confirm pivots, so those tests use (1,1) for daily_external too (the
# ARM/entry mechanics aren't what's under test there).
SIMPLE_CFG = {
    **CFG,
    "entry_exit_strategy": {
        **CFG["entry_exit_strategy"],
        "daily_external_structure": {"left_bars": 1, "right_bars": 1},
    },
}


def _bullish_weekly_df(periods: int = 6) -> pd.DataFrame:
    # same values as tests/fixtures/synthetic_uptrend_bos.csv (proven in
    # test_bos_choch.py to produce a bullish BOS at bar5 with left=1/right=1
    # pivots) — needs at least left+right+1 bars per bar to detect any
    # pivot at all, so this can't be shrunk further
    index = pd.date_range("2025-12-07", periods=periods, freq="W", tz="UTC")
    return pd.DataFrame(
        {
            "open": [1.00, 0.99, 0.98, 1.01, 1.04, 1.02][:periods],
            "high": [1.00, 1.00, 1.02, 1.05, 1.03, 1.08][:periods],
            "low": [0.98, 0.97, 0.99, 1.00, 1.01, 1.015][:periods],
            "close": [0.99, 0.98, 1.01, 1.04, 1.02, 1.07][:periods],
        },
        index=index,
    )


def _bearish_then_bullish_choch_df(periods: int) -> pd.DataFrame:
    # mirror image (2.0 - value, high/low swapped) of tests/fixtures/synthetic_choch_reversal.csv,
    # which is proven (test_bos_choch.py) to produce BOS at bar5 then CHoCH
    # at bar7 in the BULLISH direction — mirroring flips both to BEARISH
    # BOS at bar5 then BULLISH CHoCH at bar7, which is exactly the
    # retracement-then-reversal shape this strategy enters on (used here
    # with a (1,1) window via SIMPLE_CFG, not the (3,3)/(1,1) external/
    # internal split — see the dedicated ARM/entry fixture below for that).
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


def _retracement_then_reversal_df(periods: int) -> pd.DataFrame:
    """Hand-verified (by running the real detect_fractal_pivots/
    compute_structure engine, not derived on paper) so that: the (3,3)
    "external" window confirms a bullish BOS at bar10 and never reverses
    through bar15; the (1,1) "internal" window confirms its OWN bearish
    CHoCH at bar14 (a genuine pullback, while external is already bullish)
    then a bullish CHoCH at bar15 (the pullback resolving) — exactly the
    "trend turns bullish, wait for retracement, enter on reversal" flow
    this strategy is built around.
    """
    close = [100.0, 96.42, 93.47, 95.77, 100.48, 105.59, 103.16, 102.08,
             102.50, 103.56, 106.88, 103.88, 106.35, 105.69, 103.07, 107.56]
    high = [c + 0.5 for c in close]
    low = [c - 0.5 for c in close]
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close},
        index=pd.date_range("2026-01-01", periods=len(close), freq="D", tz="UTC"),
    )
    return df.iloc[:periods]


def test_flat_stays_flat_before_the_retracement_bar():
    weekly = _bullish_weekly_df()
    daily = _retracement_then_reversal_df(14)  # before bar14's internal bearish CHoCH

    result = evaluate_position_transition(weekly, daily, CFG, dict(FLAT_POSITION))

    assert result["new_position"]["state"] == "FLAT"
    assert result["event"] is None


def test_flat_arms_on_internal_bearish_reversal_while_external_bullish():
    weekly = _bullish_weekly_df()
    daily = _retracement_then_reversal_df(15)  # includes bar14's internal bearish CHoCH

    result = evaluate_position_transition(weekly, daily, CFG, dict(FLAT_POSITION))

    assert result["new_position"]["state"] == "ARMED"
    assert result["event"]["type"] == "armed"


def test_armed_fires_entry_signal_on_internal_bullish_choch():
    weekly = _bullish_weekly_df()
    daily = _retracement_then_reversal_df(16)  # includes bar15's internal bullish CHoCH
    armed = {**FLAT_POSITION, "state": "ARMED"}

    result = evaluate_position_transition(weekly, daily, CFG, armed)

    assert result["new_position"]["state"] == "PENDING_ENTRY"
    assert result["event"]["type"] == "signal_entry"
    assert result["new_position"]["stop_price"] is not None


def test_armed_disarms_if_external_trend_stops_being_bullish():
    weekly = _bullish_weekly_df()
    # truncate to bar13, then swap in a position that's ARMED (as if armed
    # yesterday) while feeding a daily series where external isn't bullish
    # yet (bar9, before the bar10 external BOS) — simulates external
    # losing its bullish status while armed
    daily = _retracement_then_reversal_df(9)
    armed = {**FLAT_POSITION, "state": "ARMED"}

    result = evaluate_position_transition(weekly, daily, CFG, armed)

    assert result["new_position"]["state"] == "FLAT"
    assert result["event"] is None


def test_pending_entry_fills_at_todays_open_not_signal_close():
    weekly = _bullish_weekly_df()
    daily = _bearish_then_bullish_choch_df(8)
    extra_day = pd.DataFrame(
        {"open": [1.10], "high": [1.12], "low": [1.08], "close": [1.11]},
        index=pd.date_range(daily.index[-1] + pd.Timedelta(days=1), periods=1, freq="D", tz="UTC"),
    )
    daily_today = pd.concat([daily, extra_day])
    pending = {**FLAT_POSITION, "state": "PENDING_ENTRY", "stop_price": 0.90}

    result = evaluate_position_transition(weekly, daily_today, SIMPLE_CFG, pending)

    assert result["new_position"]["state"] == "LONG"
    assert result["event"]["type"] == "entered"
    assert result["event"]["entry_price"] == 1.10  # today's OPEN, not close
    assert result["new_position"]["stop_price"] == 0.90  # carried over unchanged
    assert result["new_position"]["partial_taken"] is False


def test_long_exits_on_stop_breach_at_stop_price_when_no_gap():
    weekly = _bullish_weekly_df()
    daily = _bearish_then_bullish_choch_df(8)
    # last bar (idx7): open=0.94, low=0.935 — today's open is already BELOW
    # a 0.97 stop, so that scenario belongs to the gap-through test below.
    # Use a stop (0.938) that sits between open and low — only the low
    # breaches it, an ordinary intraday stop-loss with no overnight gap.
    position = {**FLAT_POSITION, "state": "LONG", "entry_price": 1.00, "entry_date": "2026-01-01", "stop_price": 0.938}
    assert daily["open"].iloc[-1] > 0.938 > daily["low"].iloc[-1]

    result = evaluate_position_transition(weekly, daily, SIMPLE_CFG, position)

    assert result["new_position"]["state"] == "FLAT"
    assert result["event"]["type"] == "exit_stop"
    assert result["event"]["exit_price"] == 0.938


def test_long_exits_at_the_gapped_down_open_when_worse_than_stop():
    weekly = _bullish_weekly_df()
    daily = _bearish_then_bullish_choch_df(8)
    position = {**FLAT_POSITION, "state": "LONG", "entry_price": 1.00, "entry_date": "2026-01-01", "stop_price": 0.97}
    assert daily["open"].iloc[-1] < 0.97
    assert daily["low"].iloc[-1] < 0.97

    result = evaluate_position_transition(weekly, daily, SIMPLE_CFG, position)

    assert result["new_position"]["state"] == "FLAT"
    assert result["event"]["type"] == "exit_stop"
    assert result["event"]["exit_price"] == daily["open"].iloc[-1]


def test_long_exits_on_daily_external_trend_reversal_not_weekly():
    # weekly stays bullish throughout (proves the exit is keyed off
    # DAILY-external, not weekly, per the redesign) while the daily series
    # itself reverses bearish (bullish BOS at bar5, bearish CHoCH at bar7 —
    # the same proven shape as tests/fixtures/synthetic_choch_reversal.csv)
    weekly = _bullish_weekly_df()
    daily = pd.DataFrame(
        {
            "open": [1.00, 0.99, 0.98, 1.01, 1.04, 1.02, 1.07, 1.06],
            "high": [1.00, 1.00, 1.02, 1.05, 1.03, 1.08, 1.075, 1.065],
            "low": [0.98, 0.97, 0.99, 1.00, 1.01, 1.015, 1.05, 0.95],
            "close": [0.99, 0.98, 1.01, 1.04, 1.02, 1.07, 1.06, 0.96],
        },
        index=pd.date_range("2026-03-02", periods=8, freq="D", tz="UTC"),
    )
    position = {**FLAT_POSITION, "state": "LONG", "entry_price": 1.00, "entry_date": "2026-01-01", "stop_price": 0.50}  # stop far away, won't trigger

    result = evaluate_position_transition(weekly, daily, SIMPLE_CFG, position)

    assert result["new_position"]["state"] == "FLAT"
    assert result["event"]["type"] == "exit_trend"


def _flat_no_pivot_df() -> pd.DataFrame:
    # Only 3 bars, deliberately shaped so neither a (1,1) pivot high nor low
    # confirms anywhere (bar1 isn't a strict extreme vs its neighbors) —
    # daily_external/internal trend both stay None (not "bearish"), so a
    # position already LONG neither gets stopped nor trend-exited here;
    # useful for isolating the stop/partial-target checks from the
    # structure-detection machinery.
    return pd.DataFrame(
        {
            "open": [1.00, 1.00, 1.02],
            "high": [1.01, 1.05, 1.25],
            "low": [0.99, 0.99, 1.01],
            "close": [1.00, 1.02, 1.20],
        },
        index=pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC"),
    )


def test_long_holds_when_no_exit_condition_met():
    weekly = _bullish_weekly_df()
    daily = _flat_no_pivot_df()
    position = {**FLAT_POSITION, "state": "LONG", "entry_price": 1.00, "entry_date": "2026-01-01", "stop_price": 0.10}

    result = evaluate_position_transition(weekly, daily, SIMPLE_CFG, position)

    assert result["new_position"]["state"] == "LONG"
    assert result["event"] is None


def test_partial_target_disabled_by_default_never_fires():
    weekly = _bullish_weekly_df()
    daily = _flat_no_pivot_df()
    position = {**FLAT_POSITION, "state": "LONG", "entry_price": 1.00, "entry_date": "2026-01-01", "stop_price": 0.10}

    # SIMPLE_CFG's partial_target.enabled is False — even though today's
    # high (1.25) is far above entry (would trivially satisfy any min_r),
    # nothing should happen
    result = evaluate_position_transition(weekly, daily, SIMPLE_CFG, position)

    assert result["new_position"]["state"] == "LONG"
    assert result["new_position"]["partial_taken"] is False
    assert result["event"] is None


def test_partial_target_fires_when_enabled_and_price_reaches_min_r():
    weekly = _bullish_weekly_df()
    daily = _flat_no_pivot_df()
    enabled_cfg = {
        **SIMPLE_CFG,
        "entry_exit_strategy": {
            **SIMPLE_CFG["entry_exit_strategy"],
            "partial_target": {"enabled": True, "min_r": 2.0, "fraction": 0.25},
        },
    }
    # entry 1.00, stop 0.90 -> risk 0.10/share -> target = 1.00 + 2*0.10 = 1.20; today's high is 1.25
    position = {**FLAT_POSITION, "state": "LONG", "entry_price": 1.00, "entry_date": "2026-01-01", "stop_price": 0.90}
    assert daily["high"].iloc[-1] == 1.25

    result = evaluate_position_transition(weekly, daily, enabled_cfg, position)

    assert result["new_position"]["state"] == "LONG"  # partial exit doesn't close the position
    assert result["new_position"]["partial_taken"] is True
    assert result["new_position"]["remaining_fraction"] == 0.75
    assert result["new_position"]["stop_price"] == 0.90  # unchanged, not moved to breakeven
    assert result["event"]["type"] == "partial_exit"
    assert result["event"]["exit_price"] == 1.20  # target itself, not the (higher) open — no gap here
