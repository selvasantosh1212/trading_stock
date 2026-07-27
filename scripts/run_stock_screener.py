"""Daily entry point for stock_hh_ll_tool. Meant to run once a day, after
NSE market close (~3:30pm IST / ~10:00 UTC), against a watchlist configured
in stock_hh_ll_tool/config.yaml.
"""

from pathlib import Path

import pandas as pd

from smc_validator.data_ingestion.resample import resample_ohlc
from stock_hh_ll_tool.config import load_config
from stock_hh_ll_tool.data import fetch_ohlc
from stock_hh_ll_tool.entry_exit import FLAT_POSITION, evaluate_position_transition
from stock_hh_ll_tool.position_state import load_positions, save_positions
from stock_hh_ll_tool.screener import screen_symbol
from stock_hh_ll_tool.telegram_notify import send_telegram_message

REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "stock_screener_latest.md"


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


def _run_entry_exit_check(symbol: str, cfg: dict, positions: dict, daily: pd.DataFrame) -> tuple[dict, str | None]:
    """Returns (updated position, a message line if something happened or
    errored, else None). Never raises — a bad ticker here shouldn't stop
    the rest of the watchlist any more than it does for the daily check.
    Takes the already-fetched `daily` bars (shared with the plain HH/LL
    check in main()) rather than re-fetching them.
    """
    try:
        weekly = resample_ohlc(daily, "W")
    except Exception as e:
        return positions.get(symbol, dict(FLAT_POSITION)), f"  entry/exit check ERROR for {symbol}: {e}"

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
            f"📥 *{symbol}*: entry signal — retracement resolved, will enter at TOMORROW's open. "
            f"Signal price {event['signal_price']:.2f}, stop {event['stop_price']:.2f} "
            f"(risk/share {risk_per_share:.2f}). Size = ({ecfg['risk_per_trade_pct']}% of capital) / {risk_per_share:.2f}."
        )
    elif event["type"] == "entered":
        msg = f"✅ *{symbol}*: entered at {event['entry_price']:.2f} (today's open). Stop at {event['stop_price']:.2f}."
    elif event["type"] == "exit_stop":
        msg = _format_exit_message(symbol, "🛑", "stopped out", event)
    elif event["type"] == "exit_trend":
        msg = _format_exit_message(symbol, "📤", "trend reversed — exited", event)
    else:
        msg = f"{symbol}: unrecognized event {event}"

    return result["new_position"], msg


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

    message_sections = []

    if flagged:
        hh_ll_lines = ["*Stock HH/LL Screener — Structure Break Confirmed Today*", ""]
        for r in flagged:
            bias_note = ""
            if r["intraday_bias"]:
                agree = r["intraday_bias"]["bias_timeframes_agree"]
                bias_note = " — ✅ intraday agrees" if agree else " — ⚠️ intraday not aligned yet"
            if r["hh_broken_today"]:
                gap_note = f" (gapped up {r['gap_pct']:+.1f}%)" if r["gapped_up_today"] else ""
            else:
                gap_note = f" (gapped down {r['gap_pct']:+.1f}%)" if r["gapped_down_today"] else ""
            hh_ll_lines.append(
                f"• *{r['symbol']}*: {r['structure_event']} ({r['structure_event_direction']}) confirmed"
                f"{gap_note}{bias_note}{_target_note(r)}"
            )
        message_sections.append("\n".join(hh_ll_lines))
        print(f"\n{len(flagged)} stock(s) flagged on the daily HH/LL check.")
        lines.append(f"{len(flagged)} stock(s) flagged on the daily HH/LL check.")
    else:
        print("\nNo structure breaks today across the watchlist.")
        lines.append("No structure breaks today across the watchlist.")

    # entry/exit strategy check (validated via reports/stock_entry_exit_strategy_investigation.md)
    # — separate, stateful, spans multiple days per position
    if entry_exit_enabled:
        positions = load_positions()
        entry_exit_messages = []
        lines.append("")
        lines.append("## Entry/Exit Strategy (Weekly/Daily)")
        for symbol in cfg["watchlist"]:
            daily = daily_by_symbol[symbol]
            if daily is None:
                continue  # already reported as a fetch error above
            _, msg = _run_entry_exit_check(symbol, cfg, positions, daily)
            if msg:
                print(msg)
                lines.append(f"- {msg}")
                entry_exit_messages.append(msg)
        save_positions(positions)
        if entry_exit_messages:
            message_sections.append("*Entry/Exit Strategy Updates*\n\n" + "\n".join(entry_exit_messages))
        else:
            lines.append("No entry/exit events today.")

    if message_sections:
        message = "\n\n".join(message_sections)
        sent = send_telegram_message(message, cfg)
        status = "Telegram message sent." if sent else (
            "Telegram NOT sent — set STOCK_TOOL_TELEGRAM_BOT_TOKEN and "
            "STOCK_TOOL_TELEGRAM_CHAT_ID (see stock_hh_ll_tool/config.yaml)."
        )
        print(f"\n{status}")
        lines.append(f"\n{status}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
