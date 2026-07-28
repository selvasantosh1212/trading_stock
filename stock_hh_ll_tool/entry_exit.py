import pandas as pd

from smc_validator.data_ingestion.resample import align_htf_to_ltf
from smc_validator.patterns.order_blocks import atr
from smc_validator.structure.bos_choch import compute_structure
from smc_validator.structure.swings import detect_fractal_pivots
from stock_hh_ll_tool.targets import next_liquidity_targets

FLAT_POSITION = {
    "state": "FLAT",
    "entry_price": None,
    "entry_date": None,
    "stop_price": None,
    "partial_taken": False,
    "partial_exit_price": None,
    "partial_exit_date": None,
    "remaining_fraction": 1.0,
}


def evaluate_position_transition(weekly_df: pd.DataFrame, daily_df: pd.DataFrame, cfg: dict, position: dict) -> dict:
    """Pure logic (no I/O) — one day's position-state transition for the
    swing-trade strategy. See the header comment above `entry_exit_strategy`
    in stock_hh_ll_tool/config.yaml for the full design rationale (a
    2026-07-27 3-agent redesign: Weekly is now a confirmation gate only, the
    daily-external trend reversal is the exit, and a daily-internal
    retracement-then-reversal times entry — plus why a liquidity-target
    exit was deliberately rejected).

    `position` is the state carried in from yesterday (or FLAT_POSITION on
    first run): {"state": "FLAT"|"ARMED"|"PENDING_ENTRY"|"LONG",
    "stop_price", "entry_price", "entry_date", "partial_taken",
    "partial_exit_price", "partial_exit_date", "remaining_fraction"}.
    Returns {"new_position": ..., "event": dict|None, "next_liquidity":
    dict} — `event` describes what just happened (for alerting/display),
    `next_liquidity` is next_liquidity_targets(...) for the daily-external
    structure computed here (INFORMATIONAL ONLY — see that function's
    docstring for why it isn't wired into the exit logic).

    State flow: FLAT -> ARMED (a retracement against the still-bullish
    trend just started) -> PENDING_ENTRY (the retracement just resolved
    bullish; fills tomorrow) -> LONG -> back to FLAT (stop or trend-exit).
    Re-entry after an exit requires arming fresh (a new retracement), so a
    stop-out can't instantly re-buy the same failed leg.
    """
    ecfg = cfg["entry_exit_strategy"]
    wk_left, wk_right = ecfg["weekly_structure"]["left_bars"], ecfg["weekly_structure"]["right_bars"]
    ext_left, ext_right = ecfg["daily_external_structure"]["left_bars"], ecfg["daily_external_structure"]["right_bars"]
    int_left, int_right = ecfg["daily_internal_structure"]["left_bars"], ecfg["daily_internal_structure"]["right_bars"]

    weekly_structure = compute_structure(weekly_df, detect_fractal_pivots(weekly_df, wk_left, wk_right), close_only=True)
    daily_ext = compute_structure(daily_df, detect_fractal_pivots(daily_df, ext_left, ext_right), close_only=True)
    daily_int = compute_structure(daily_df, detect_fractal_pivots(daily_df, int_left, int_right), close_only=True)
    weekly_bias_on_daily = align_htf_to_ltf(weekly_structure["trend"], daily_df.index)

    today = {
        "weekly_bias": weekly_bias_on_daily.iloc[-1],
        "ext_trend": daily_ext["trend"].iloc[-1],
        "int_event": daily_int["event"].iloc[-1],
        "int_event_dir": daily_int["event_direction"].iloc[-1],
        "int_swing_low": daily_int["swing_low"].iloc[-1],
        "open": float(daily_df["open"].iloc[-1]),
        "high": float(daily_df["high"].iloc[-1]),
        "low": float(daily_df["low"].iloc[-1]),
        "close": float(daily_df["close"].iloc[-1]),
        "date": str(daily_df.index[-1].date()),
    }
    bullish_permission = today["weekly_bias"] == "bullish" and today["ext_trend"] == "bullish"
    state = position.get("state", "FLAT")

    result = _transition(state, position, bullish_permission, today, daily_df, ecfg)
    result["next_liquidity"] = next_liquidity_targets(daily_df, daily_ext)
    return result


def _transition(state: str, position: dict, bullish_permission: bool, today: dict, daily_df: pd.DataFrame, ecfg: dict) -> dict:
    if state == "PENDING_ENTRY":
        new_position = {
            **FLAT_POSITION,
            "state": "LONG",
            "entry_price": today["open"],
            "entry_date": today["date"],
            "stop_price": position["stop_price"],
        }
        event = {"type": "entered", "entry_price": today["open"], "stop_price": position["stop_price"]}
        return {"new_position": new_position, "event": event}

    if state == "LONG":
        stop_price = position["stop_price"]

        # stop-first on a same-bar stop-and-target ambiguity (the more
        # conservative resolution — matches the investigation's approach)
        if stop_price is not None and today["low"] <= stop_price:
            exit_price = today["open"] if today["open"] < stop_price else stop_price
            event = {
                "type": "exit_stop",
                "exit_price": exit_price,
                "entry_price": position["entry_price"],
                "partial_exit_price": position.get("partial_exit_price"),
                "remaining_fraction": position.get("remaining_fraction", 1.0),
            }
            return {"new_position": dict(FLAT_POSITION), "event": event}

        if today["ext_trend"] == "bearish":
            event = {
                "type": "exit_trend",
                "exit_price": today["close"],
                "entry_price": position["entry_price"],
                "partial_exit_price": position.get("partial_exit_price"),
                "remaining_fraction": position.get("remaining_fraction", 1.0),
            }
            return {"new_position": dict(FLAT_POSITION), "event": event}

        pcfg = ecfg.get("partial_target", {})
        if pcfg.get("enabled") and not position.get("partial_taken") and stop_price is not None:
            risk_per_share = position["entry_price"] - stop_price
            target_price = position["entry_price"] + pcfg["min_r"] * risk_per_share
            if risk_per_share > 0 and today["high"] >= target_price:
                fill_price = today["open"] if today["open"] > target_price else target_price
                new_position = {
                    **position,
                    "partial_taken": True,
                    "partial_exit_price": fill_price,
                    "partial_exit_date": today["date"],
                    "remaining_fraction": 1.0 - pcfg["fraction"],
                }
                event = {"type": "partial_exit", "exit_price": fill_price, "fraction": pcfg["fraction"], "entry_price": position["entry_price"]}
                return {"new_position": new_position, "event": event}

        return {"new_position": position, "event": None}

    if state == "ARMED":
        if not bullish_permission:
            return {"new_position": dict(FLAT_POSITION), "event": None}
        if today["int_event"] == "CHoCH" and today["int_event_dir"] == "bullish" and pd.notna(today["int_swing_low"]):
            atr_today = atr(daily_df).iloc[-1]
            buffer = ecfg["stop_atr_buffer_fraction"] * atr_today if pd.notna(atr_today) else 0.0
            stop_price = float(today["int_swing_low"]) - buffer
            new_position = {**FLAT_POSITION, "state": "PENDING_ENTRY", "stop_price": stop_price}
            event = {"type": "signal_entry", "signal_price": today["close"], "stop_price": stop_price}
            return {"new_position": new_position, "event": event}
        # a fresh bearish internal event just deepens the retracement —
        # stays armed either way, no event (avoid alerting on every wiggle)
        return {"new_position": position, "event": None}

    # FLAT: arm on a retracement starting — a bearish internal reversal
    # while the bigger trend (weekly + daily-external) is already bullish.
    # pd.notna (not `is not None`): once daily_int's "event" column has seen
    # ANY real event string in its history, pandas 3.0's string dtype
    # represents the OTHER missing cells as float NaN, not None (verified
    # directly — a column of all-None stays object/None, but real data over
    # a multi-year history will essentially always have at least one event).
    if bullish_permission and pd.notna(today["int_event"]) and today["int_event_dir"] == "bearish":
        event = {"type": "armed"}
        return {"new_position": {**FLAT_POSITION, "state": "ARMED"}, "event": event}

    return {"new_position": dict(FLAT_POSITION), "event": None}
