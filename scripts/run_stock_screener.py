"""Daily entry point for stock_hh_ll_tool. Meant to run once a day, after
NSE market close (~3:30pm IST / ~10:00 UTC), against a watchlist configured
in stock_hh_ll_tool/config.yaml. Writes results to docs/data.json, read by
the static dashboard (docs/index.html) served over GitHub Pages — replaces
the Telegram notification path, which the user dropped due to unrelated
Telegram spam making alerts hard to notice.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from smc_validator.data_ingestion.resample import resample_ohlc
from stock_hh_ll_tool.config import load_config
from stock_hh_ll_tool.data import fetch_ohlc
from stock_hh_ll_tool.entry_exit import FLAT_POSITION, evaluate_position_transition
from stock_hh_ll_tool.position_state import load_positions, save_positions
from stock_hh_ll_tool.screener import screen_symbol

REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "stock_screener_latest.md"
DASHBOARD_DATA_PATH = Path(__file__).resolve().parent.parent / "docs" / "data.json"


def _json_safe(obj):
    """Recursively replaces float('nan') with None. Needed because pandas'
    (3.0+) string-dtype columns surface missing cells as a float nan scalar
    via .iloc rather than None — harmless for the existing `== "CHoCH"`-style
    comparisons elsewhere (NaN just compares False), but json.dumps emits a
    bare `NaN` token for it, which isn't valid JSON and breaks the
    dashboard's JSON.parse.
    """
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _target_note(r: dict) -> str:
    target = r.get("next_target")
    if not target:
        return ""
    if target["next_target"] is None:
        return f" | next level: {target['note']}"
    return f" | next level: {target['next_target']:.2f} ({target['next_target_pct']:+.1f}%)"


def _format_exit_message(symbol: str, emoji: str, verb: str, event: dict) -> str:
    pnl_pct = (event["exit_price"] / event["entry_price"] - 1) * 100
    return f"{emoji} *{symbol}*: {verb} at {event['exit_price']:.2f} (entry was {event['entry_price']:.2f}, {pnl_pct:+.1f}%)."


def _run_entry_exit_check(symbol: str, cfg: dict, positions: dict, daily: pd.DataFrame) -> tuple[dict, dict | None]:
    """Returns (updated position, an event dict {"type", "message"} if
    something happened or errored, else None). Never raises — a bad ticker
    here shouldn't stop the rest of the watchlist any more than it does for
    the daily check. Takes the already-fetched `daily` bars (shared with the
    plain HH/LL check in main()) rather than re-fetching them.
    """
    try:
        weekly = resample_ohlc(daily, "W")
    except Exception as e:
        return positions.get(symbol, dict(FLAT_POSITION)), {"type": "error", "message": f"entry/exit check ERROR for {symbol}: {e}"}

    current = positions.get(symbol, dict(FLAT_POSITION))
    result = evaluate_position_transition(weekly, daily, cfg, current)
    positions[symbol] = result["new_position"]

    event = result["event"]
    if event is None:
        return result["new_position"], None

    ecfg = cfg["entry_exit_strategy"]
    if event["type"] == "signal_entry":
        risk_per_share = event["signal_price"] - event["stop_price"]
        msg = (
            f"📥 {symbol}: entry signal — retracement resolved, will enter at TOMORROW's open. "
            f"Signal price {event['signal_price']:.2f}, stop {event['stop_price']:.2f} "
            f"(risk/share {risk_per_share:.2f}). Size = ({ecfg['risk_per_trade_pct']}% of capital) / {risk_per_share:.2f}."
        )
    elif event["type"] == "entered":
        msg = f"✅ {symbol}: entered at {event['entry_price']:.2f} (today's open). Stop at {event['stop_price']:.2f}."
    elif event["type"] == "exit_stop":
        msg = _format_exit_message(symbol, "🛑", "stopped out", event)
    elif event["type"] == "exit_trend":
        msg = _format_exit_message(symbol, "📤", "trend reversed — exited", event)
    else:
        msg = f"{symbol}: unrecognized event {event}"

    return result["new_position"], {"type": event["type"], "message": msg}


def main() -> None:
    cfg = load_config()
    entry_exit_enabled = cfg.get("entry_exit_strategy", {}).get("enabled")

    # fetch each symbol's daily bars once and share them across both checks
    # below, rather than each independently re-fetching the same data
    daily_by_symbol: dict[str, pd.DataFrame | None] = {}
    fetch_errors: dict[str, str] = {}
    for symbol in cfg["watchlist"]:
        try:
            daily_by_symbol[symbol] = fetch_ohlc(symbol, "1d")
        except Exception as e:
            daily_by_symbol[symbol] = None
            fetch_errors[symbol] = str(e)

    results = [
        {"symbol": symbol, "error": fetch_errors[symbol]}
        if symbol in fetch_errors
        else screen_symbol(symbol, cfg, daily=daily_by_symbol[symbol])
        for symbol in cfg["watchlist"]
    ]

    lines = ["# Stock HH/LL Screener — Latest Run", ""]
    for r in results:
        if r.get("error"):
            line = f"- **{r['symbol']}**: ERROR — {r['error']}"
        else:
            if r["hh_broken_today"]:
                flag = "🟢 HIGHER HIGH BROKEN TODAY"
            elif r["ll_broken_today"]:
                flag = "🔴 LOWER LOW BROKEN TODAY"
            else:
                flag = "—"
            gap = f" | gap {r['gap_pct']:+.2f}%" if r["gap_pct"] is not None else ""
            line = f"- **{r['symbol']}**: {flag} | trend={r['daily_trend']} | close={r['close']}{gap}{_target_note(r)}"
        print(line)
        lines.append(line)

    flagged = [r for r in results if r.get("hh_broken_today") or r.get("ll_broken_today")]
    lines.append("")

    if flagged:
        print(f"\n{len(flagged)} stock(s) flagged on the daily HH/LL check.")
        lines.append(f"{len(flagged)} stock(s) flagged on the daily HH/LL check.")
    else:
        print("\nNo structure breaks today across the watchlist.")
        lines.append("No structure breaks today across the watchlist.")

    # entry/exit strategy check (validated via reports/stock_entry_exit_strategy_investigation.md)
    # — separate, stateful, spans multiple days per position
    entry_exit_events = []
    positions: dict = {}
    if entry_exit_enabled:
        positions = load_positions()
        lines.append("")
        lines.append("## Entry/Exit Strategy (Weekly/Daily)")
        for symbol in cfg["watchlist"]:
            daily = daily_by_symbol[symbol]
            if daily is None:
                continue  # already reported as a fetch error above
            _, event = _run_entry_exit_check(symbol, cfg, positions, daily)
            if event:
                print(event["message"])
                lines.append(f"- {event['message']}")
                entry_exit_events.append({"symbol": symbol, **event})
        save_positions(positions)
        if not entry_exit_events:
            lines.append("No entry/exit events today.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))

    dashboard = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stocks": results,
        "flagged_count": len(flagged),
        "entry_exit_events": entry_exit_events,
        "positions": positions,
    }
    DASHBOARD_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD_DATA_PATH.write_text(json.dumps(_json_safe(dashboard), indent=2))
    print(f"\nDashboard data written to {DASHBOARD_DATA_PATH}")


if __name__ == "__main__":
    main()
