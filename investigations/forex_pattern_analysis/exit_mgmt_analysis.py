"""Characterize WINNING trades' exit/management behavior for the validated
151-trade EURUSD SMC backtest. Read-only reuse of smc_validator/ and
scripts/run_backtest.py -- no files under those dirs are modified. Uses a
local instrumented copy of simulate_trades (exit_mgmt_simulate.py) purely to
capture exit-bar timestamps, which the original engine does not return.

Outputs:
- exit_mgmt_trades.csv          (all 151 trades, full instrumented fields)
- exit_mgmt_winners.csv         (winning subset only)
- exit_mgmt_report.txt          (printed report, also saved)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.run_backtest import build_signals  # noqa: E402
from smc_validator.config import load_config  # noqa: E402

from exit_mgmt_simulate import simulate_trades_instrumented  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    cfg = load_config()
    bars, signals = build_signals(cfg)
    print(f"Bars: {len(bars)}, candidate signals: {len(signals)}")

    trades = simulate_trades_instrumented(bars, signals, staged_exit_split=tuple(cfg["risk"]["staged_exit_split"]))
    print(f"Simulated trades: {len(trades)}")

    trades.to_csv(OUT_DIR / "exit_mgmt_trades.csv", index=False)

    winners = trades[trades["total_r"] > 0].copy()
    winners.to_csv(OUT_DIR / "exit_mgmt_winners.csv", index=False)

    n_total = len(trades)
    n_win = len(winners)
    win_rate = n_win / n_total

    lines = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("=" * 70)
    emit("EXIT/MANAGEMENT CHARACTERISTICS OF WINNING TRADES")
    emit("=" * 70)
    emit(f"Total simulated trades: {n_total}")
    emit(f"Winning trades (total_r > 0): {n_win}  ({win_rate:.1%} win rate)")
    emit("")

    # ---- 1. Exit reason distribution among winners ----
    emit("-" * 70)
    emit("1. EXIT REASON DISTRIBUTION (winners only)")
    emit("-" * 70)
    reason_counts = winners["exit_reason"].value_counts()
    reason_pct = winners["exit_reason"].value_counts(normalize=True)
    for reason in reason_counts.index:
        emit(f"  {reason:28s}: {reason_counts[reason]:4d} trades ({reason_pct[reason]:.1%} of winners)")

    both_targets = winners[winners["exit_reason"] == "both_targets"]
    stop_after_partial = winners[winners["exit_reason"] == "stop_after_partial"]
    emit("")
    emit(
        f"  'Clean' full-target wins (both_targets): {len(both_targets)} "
        f"({len(both_targets) / n_win:.1%} of winners)"
    )
    emit(
        f"  'Partial-credit' wins (stop_after_partial, still net positive): "
        f"{len(stop_after_partial)} ({len(stop_after_partial) / n_win:.1%} of winners)"
    )
    if len(stop_after_partial) > 0:
        emit(
            f"    -> avg total_r for stop_after_partial winners: "
            f"{stop_after_partial['total_r'].mean():.3f}R  vs both_targets winners: "
            f"{both_targets['total_r'].mean():.3f}R"
        )
    other_reasons = winners[~winners["exit_reason"].isin(["both_targets", "stop_after_partial"])]
    if len(other_reasons) > 0:
        emit(f"  Other exit reasons among winners: {other_reasons['exit_reason'].value_counts().to_dict()}")
    emit("")

    # ---- 2. leg1_r vs leg2_r contribution ----
    emit("-" * 70)
    emit("2. LEG1 vs LEG2 CONTRIBUTION (winners only)")
    emit("-" * 70)
    split0, split1 = cfg["risk"]["staged_exit_split"]
    winners["leg1_contrib"] = split0 * winners["leg1_r"]
    # leg2_r is None (NaN) for exits before target1 was ever hit (shouldn't
    # happen among winners since leg1_r would be -1 then, i.e. a loss) --
    # but guard anyway: use leg1_r as leg2 fallback exactly like the engine
    # does when leg2_r is NaN (this only occurs for unresolved/end-of-data
    # style trades, not for actual winners with both_targets/stop_after_partial)
    leg2_effective = winners["leg2_r"].where(winners["leg2_r"].notna(), winners["leg1_r"])
    winners["leg2_contrib"] = split1 * leg2_effective

    emit(f"  leg1_r  (winners): mean={winners['leg1_r'].mean():.3f}R  median={winners['leg1_r'].median():.3f}R")
    have_leg2 = winners["leg2_r"].notna()
    emit(
        f"  leg2_r  (winners, where applicable, n={have_leg2.sum()}): "
        f"mean={winners.loc[have_leg2, 'leg2_r'].mean():.3f}R  "
        f"median={winners.loc[have_leg2, 'leg2_r'].median():.3f}R"
    )
    emit(f"    (leg2_r is negative for stop_after_partial winners -- it's the losing leg that still nets positive)")
    emit("")
    total_leg1_contrib = winners["leg1_contrib"].sum()
    total_leg2_contrib = winners["leg2_contrib"].sum()
    total_r_sum = winners["total_r"].sum()
    emit(f"  Aggregate winning R: {total_r_sum:.2f}R total across {n_win} winners")
    emit(
        f"    leg1 contribution: {total_leg1_contrib:.2f}R "
        f"({total_leg1_contrib / total_r_sum:.1%} of total winning R)"
    )
    emit(
        f"    leg2 contribution: {total_leg2_contrib:.2f}R "
        f"({total_leg2_contrib / total_r_sum:.1%} of total winning R)"
    )
    emit(
        "  -> leg1 (the nearer, staged target1) is the dominant, more reliable"
        " driver of winning R; leg2 (holding for target2) is a smaller net add"
        " because it's dragged down by the stop_after_partial subset."
    )
    # breakdown by exit reason for context
    emit("")
    emit("  Leg contribution restricted to both_targets winners only (both legs positive):")
    if len(both_targets) > 0:
        bt_leg1 = (split0 * both_targets["leg1_r"]).sum()
        bt_leg2 = (split1 * both_targets["leg2_r"]).sum()
        bt_total = both_targets["total_r"].sum()
        emit(f"    leg1: {bt_leg1:.2f}R ({bt_leg1 / bt_total:.1%})   leg2: {bt_leg2:.2f}R ({bt_leg2 / bt_total:.1%})")
    emit("")

    # ---- 3. Holding duration ----
    emit("-" * 70)
    emit("3. HOLDING DURATION (winners only, signal_time -> exit_time)")
    emit("-" * 70)
    winners["elapsed"] = winners["exit_time"] - winners["signal_time"]
    winners["elapsed_hours"] = winners["elapsed"].dt.total_seconds() / 3600.0
    bar_minutes = 5
    winners["bars_calendar"] = (winners["elapsed"].dt.total_seconds() / 60.0 / bar_minutes).round().astype(int)

    emit("  In 5-minute BAR COUNTS (exit_bars = bars walked forward, includes any partial-fill bar):")
    emit(
        f"    mean={winners['exit_bars'].mean():.1f} bars  median={winners['exit_bars'].median():.0f} bars  "
        f"max={winners['exit_bars'].max():.0f} bars  min={winners['exit_bars'].min():.0f} bars"
    )
    emit("  In ELAPSED CALENDAR TIME (includes overnight/weekend gaps where market was closed):")
    emit(
        f"    mean={winners['elapsed_hours'].mean():.1f}h  median={winners['elapsed_hours'].median():.1f}h  "
        f"max={winners['elapsed_hours'].max():.1f}h ({winners['elapsed_hours'].max() / 24:.1f} days)  "
        f"min={winners['elapsed_hours'].min():.1f}h"
    )
    emit("")
    emit("  By exit reason:")
    for reason, grp in winners.groupby("exit_reason"):
        emit(
            f"    {reason:20s} n={len(grp):3d}  mean_bars={grp['exit_bars'].mean():.1f}  "
            f"median_bars={grp['exit_bars'].median():.0f}  mean_hours={grp['elapsed_hours'].mean():.1f}h"
        )
    emit("")
    # time to just leg1 (target1) for context
    have_t1 = winners["t1_hit_bars"].notna()
    if have_t1.sum() > 0:
        emit(
            f"  Time to reach target1 alone (n={have_t1.sum()}): "
            f"mean={winners.loc[have_t1, 't1_hit_bars'].mean():.1f} bars, "
            f"median={winners.loc[have_t1, 't1_hit_bars'].median():.0f} bars"
        )
    emit("")

    # ---- 4. Direction split ----
    emit("-" * 70)
    emit("4. DIRECTION SPLIT: winners vs overall")
    emit("-" * 70)
    overall_dir = trades["direction"].value_counts(normalize=True)
    winner_dir = winners["direction"].value_counts(normalize=True)
    overall_dir_n = trades["direction"].value_counts()
    winner_dir_n = winners["direction"].value_counts()
    for d in ["bullish", "bearish"]:
        o_n = overall_dir_n.get(d, 0)
        w_n = winner_dir_n.get(d, 0)
        o_pct = overall_dir.get(d, 0.0)
        w_pct = winner_dir.get(d, 0.0)
        win_rate_within_dir = w_n / o_n if o_n > 0 else float("nan")
        emit(
            f"  {d:8s}: overall {o_n:3d}/{n_total} ({o_pct:.1%})  |  winners {w_n:3d}/{n_win} ({w_pct:.1%})  |  "
            f"win rate WITHIN {d}: {win_rate_within_dir:.1%}"
        )
    emit("")
    lift_bullish = (winner_dir.get("bullish", 0) - overall_dir.get("bullish", 0)) * 100
    emit(
        f"  Representation shift: bullish share of winners is "
        f"{lift_bullish:+.1f} percentage points vs its share of all candidate trades."
    )

    emit("")
    emit("=" * 70)

    (OUT_DIR / "exit_mgmt_report.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
