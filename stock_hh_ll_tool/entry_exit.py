import pandas as pd

from smc_validator.data_ingestion.resample import align_htf_to_ltf
from smc_validator.patterns.order_blocks import atr
from smc_validator.structure.bos_choch import compute_structure
from smc_validator.structure.swings import detect_fractal_pivots

FLAT_POSITION = {"state": "FLAT", "entry_price": None, "entry_date": None, "stop_price": None}


def evaluate_position_transition(weekly_df: pd.DataFrame, daily_df: pd.DataFrame, cfg: dict, position: dict) -> dict:
    """Pure logic (no I/O) — one day's position-state transition for the
    Weekly(HTF)/Daily(LTF) entry/exit strategy validated in
    reports/stock_entry_exit_strategy_investigation.md.

    `position` is the state carried in from yesterday (or FLAT_POSITION on
    first run): {"state": "FLAT"|"PENDING_ENTRY"|"LONG", "stop_price",
    "entry_price", "entry_date"}. Returns {"new_position": ..., "event":
    dict|None} — `event` describes what just happened, for alerting.

    Realistic fill: a signal detected today only becomes a position
    tomorrow, filled at tomorrow's actual open — never the signal bar's own
    close (the stress-test agent flagged same-bar-close fills as
    unrealistic for a live tool). This is why entry has two steps
    (PENDING_ENTRY -> LONG) instead of one.

    Exit fires on whichever condition is met first: the stop-loss (LTF
    swing low at entry, minus a small ATR buffer) being breached, or the
    HTF confirming a bearish reversal (trend-following exit). The
    investigation found the stop-loss variant roughly halves worst-case
    single-trade loss while slightly improving total return versus no stop
    or a fixed-percentage stop.
    """
    ecfg = cfg["entry_exit_strategy"]
    htf_left, htf_right = ecfg["htf_structure"]["left_bars"], ecfg["htf_structure"]["right_bars"]
    ltf_left, ltf_right = ecfg["ltf_structure"]["left_bars"], ecfg["ltf_structure"]["right_bars"]

    htf_structure = compute_structure(weekly_df, detect_fractal_pivots(weekly_df, htf_left, htf_right), close_only=True)
    ltf_structure = compute_structure(daily_df, detect_fractal_pivots(daily_df, ltf_left, ltf_right), close_only=True)
    htf_trend_on_ltf = align_htf_to_ltf(htf_structure["trend"], daily_df.index)

    today_htf_bias = htf_trend_on_ltf.iloc[-1]
    today_ltf_event = ltf_structure["event"].iloc[-1]
    today_ltf_event_dir = ltf_structure["event_direction"].iloc[-1]
    today_open = float(daily_df["open"].iloc[-1])
    today_low = float(daily_df["low"].iloc[-1])
    today_close = float(daily_df["close"].iloc[-1])
    today_date = str(daily_df.index[-1].date())

    state = position.get("state", "FLAT")

    if state == "PENDING_ENTRY":
        new_position = {
            "state": "LONG",
            "entry_price": today_open,
            "entry_date": today_date,
            "stop_price": position["stop_price"],
        }
        event = {"type": "entered", "entry_price": today_open, "stop_price": position["stop_price"]}
        return {"new_position": new_position, "event": event}

    if state == "LONG":
        stop_price = position["stop_price"]
        if stop_price is not None and today_low <= stop_price:
            event = {"type": "exit_stop", "exit_price": stop_price, "entry_price": position["entry_price"]}
            return {"new_position": dict(FLAT_POSITION), "event": event}
        if today_htf_bias == "bearish":
            event = {"type": "exit_trend", "exit_price": today_close, "entry_price": position["entry_price"]}
            return {"new_position": dict(FLAT_POSITION), "event": event}
        return {"new_position": position, "event": None}

    # FLAT: look for a fresh entry signal — HTF bullish AND the LTF just
    # reversed bullish (CHoCH), i.e. the retracement just resolved
    if today_htf_bias == "bullish" and today_ltf_event == "CHoCH" and today_ltf_event_dir == "bullish":
        ltf_swing_low = ltf_structure["swing_low"].iloc[-1]
        if pd.notna(ltf_swing_low):
            atr_today = atr(daily_df).iloc[-1]
            buffer = ecfg["stop_atr_buffer_fraction"] * atr_today if pd.notna(atr_today) else 0.0
            stop_price = float(ltf_swing_low) - buffer
            new_position = {"state": "PENDING_ENTRY", "stop_price": stop_price, "entry_price": None, "entry_date": None}
            event = {"type": "signal_entry", "signal_price": today_close, "stop_price": stop_price}
            return {"new_position": new_position, "event": event}

    return {"new_position": dict(FLAT_POSITION), "event": None}
