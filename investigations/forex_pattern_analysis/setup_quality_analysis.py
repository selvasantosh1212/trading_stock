"""Setup-quality analysis of the 151-trade SMC EURUSD backtest's WINNING
subset. Re-derives 15-minute ATR / FVG / order-block intermediates (which
build_signals computes internally but does not return) and joins them back
onto each winning signal's timestamp, to characterize:

  1. risk:reward ratio at entry (target1_R, target2_R) — direct from signals
  2. FVG size relative to 15m ATR, for the FVG associated with each zone
  3. order-block "balance" tightness: OB candle body / ATR
  4. confirmation speed: 5m bars between zone activation and LTF confirmation

Read-only w.r.t. smc_validator/ and scripts/ — only imports/reuses them.
Outputs CSVs + a markdown report into this same investigations subfolder.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_backtest import build_signals
from smc_validator.backtest.engine import simulate_trades
from smc_validator.config import load_config
from smc_validator.data_ingestion.resample import align_htf_to_ltf, resample_ohlc
from smc_validator.patterns.fvg import detect_fvg
from smc_validator.patterns.order_blocks import atr as compute_atr
from smc_validator.patterns.order_blocks import detect_order_blocks
from smc_validator.structure.bos_choch import compute_structure
from smc_validator.structure.swings import detect_fractal_pivots

OUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    cfg = load_config()
    bars, signals = build_signals(cfg)
    trades = simulate_trades(bars, signals, staged_exit_split=tuple(cfg["risk"]["staged_exit_split"]))

    print(f"Signals: {len(signals)}  Trades: {len(trades)}")

    trades = trades.set_index("signal_time")
    trades["is_win"] = trades["total_r"] > 0
    print(f"Wins: {trades['is_win'].sum()}  Win rate: {trades['is_win'].mean():.1%}")

    # ---- Dimension 1: risk:reward ratio at entry (direct from signals) ----
    joined = signals.join(trades[["total_r", "is_win", "exit_reason"]], how="inner")
    joined["target1_R"] = (joined["target1"] - joined["entry"]).abs() / (joined["entry"] - joined["stop"]).abs()
    joined["target2_R"] = (joined["target2"] - joined["entry"]).abs() / (joined["entry"] - joined["stop"]).abs()

    # ---- Re-derive 15m ATR / FVG / structure / order blocks (copied from
    # scripts/run_backtest.py::build_signals, since that function only
    # returns final bars/signals, not these intermediates) ----
    df_5m = bars
    df_15m = resample_ohlc(df_5m, "15min")
    atr_15m = compute_atr(df_15m)
    fvg_15m = detect_fvg(df_15m, min_gap_size=cfg["fair_value_gaps"]["min_gap_size_atr_fraction"] * atr_15m.fillna(0))
    structure_15m = compute_structure(
        df_15m,
        detect_fractal_pivots(df_15m, cfg["structure"]["external"]["left_bars"], cfg["structure"]["external"]["right_bars"]),
        close_only=cfg["structure"]["close_only_break"],
    )
    LOOKAHEAD = 5
    ob_15m = detect_order_blocks(
        df_15m,
        atr_15m,
        balance_body_atr_fraction=cfg["order_blocks"]["balance_candle_body_atr_fraction"],
        expansion_atr_multiple=cfg["order_blocks"]["expansion_move_atr_multiple"],
        lookahead=LOOKAHEAD,
        fvg_flags=fvg_15m if cfg["order_blocks"]["require_associated_fvg"] else None,
        structure_event=structure_15m["event"],
        structure_event_direction=structure_15m["event_direction"],
    )

    zone_direction_15m = pd.Series(pd.NA, index=df_15m.index, dtype="object")
    zone_direction_15m[ob_15m["bullish_ob"]] = "bullish"
    zone_direction_15m[ob_15m["bearish_ob"]] = "bearish"
    zone_start_15m = ob_15m["bullish_ob"] | ob_15m["bearish_ob"]

    timeout = cfg["ltf_confirmation"]["timeout_bars"]
    zone_active_15m = pd.Series(False, index=df_15m.index)
    zone_start_time_15m = pd.Series(pd.NaT, index=df_15m.index, dtype="datetime64[ns, UTC]" if df_15m.index.tz else "datetime64[ns]")
    bars_since_zone_start = None
    current_start_time = None
    for i in range(len(df_15m)):
        if zone_start_15m.iloc[i]:
            bars_since_zone_start = 0
            current_start_time = df_15m.index[i]
        elif bars_since_zone_start is not None:
            bars_since_zone_start += 1
        if bars_since_zone_start is not None and bars_since_zone_start <= timeout:
            zone_active_15m.iloc[i] = True
            zone_start_time_15m.iloc[i] = current_start_time

    # forward-fill the "which zone is active" origin timestamp onto 5m, same
    # asof-backward mechanism build_signals uses for zone_active_5m/zone_direction_5m
    zone_start_time_5m = align_htf_to_ltf(zone_start_time_15m, df_5m.index)

    # body/ATR at each 15m bar (tightness proxy), independent of OB status
    body_15m = (df_15m["close"] - df_15m["open"]).abs()
    body_over_atr_15m = body_15m / atr_15m

    # For each OB bar, find the associated FVG (the fvg_bullish/bearish flag
    # that made detect_order_blocks' fvg_bullish_ahead/fvg_bearish_ahead True
    # within the same forward lookahead window [i, i+LOOKAHEAD-1]) and record
    # its gap size relative to ATR at the OB bar.
    n15 = len(df_15m)
    fvg_gap = pd.Series(np.where(fvg_15m["fvg_bullish"], fvg_15m["fvg_top"] - fvg_15m["fvg_bottom"],
                                  np.where(fvg_15m["fvg_bearish"], fvg_15m["fvg_top"] - fvg_15m["fvg_bottom"], np.nan)),
                        index=df_15m.index)

    def associated_fvg_size_over_atr(ob_pos: int, direction: str) -> float | None:
        flag_col = "fvg_bullish" if direction == "bullish" else "fvg_bearish"
        end = min(ob_pos + LOOKAHEAD, n15)
        window_flags = fvg_15m[flag_col].iloc[ob_pos:end]
        matches = window_flags[window_flags].index
        if len(matches) == 0:
            return None
        first_match = matches[0]
        gap = fvg_gap.loc[first_match]
        atr_at_ob = atr_15m.iloc[ob_pos]
        if pd.isna(gap) or pd.isna(atr_at_ob) or atr_at_ob == 0:
            return None
        return float(gap / atr_at_ob)

    ob_pos_by_time = {ts: pos for pos, ts in enumerate(df_15m.index)}
    pos_by_5m_time = {ts: pos for pos, ts in enumerate(df_5m.index)}

    # ---- Join per-winning(and losing, for contrast) trade zone-level stats ----
    rows = []
    for sig_time, row in joined.iterrows():
        zone_start_ts = zone_start_time_5m.loc[sig_time] if sig_time in zone_start_time_5m.index else pd.NaT
        rec = {
            "signal_time": sig_time,
            "direction": row["direction"] if "direction" in row else None,
            "is_win": row["is_win"],
            "total_r": row["total_r"],
            "target1_R": row["target1_R"],
            "target2_R": row["target2_R"],
            "zone_start_time": zone_start_ts,
        }
        if pd.notna(zone_start_ts) and zone_start_ts in ob_pos_by_time:
            ob_pos = ob_pos_by_time[zone_start_ts]
            ob_direction = "bullish" if ob_15m["bullish_ob"].iloc[ob_pos] else (
                "bearish" if ob_15m["bearish_ob"].iloc[ob_pos] else None
            )
            rec["ob_direction"] = ob_direction
            rec["ob_body_over_atr"] = body_over_atr_15m.iloc[ob_pos]
            if ob_direction is not None:
                rec["fvg_size_over_atr"] = associated_fvg_size_over_atr(ob_pos, ob_direction)
            # Bar-position difference, NOT calendar-time / 5min — the raw 5m
            # series has real gaps (weekends, holidays) where no bars exist,
            # so dividing elapsed wall-clock time by 5 minutes would hugely
            # inflate "bars elapsed" across any such gap. zone_start_ts is a
            # 15m-bar CLOSE timestamp (may not itself be a 5m bar open); find
            # the first 5m bar at/after it as the zone's own bar-equivalent origin.
            sig_pos = pos_by_5m_time.get(sig_time)
            origin_pos = int(df_5m.index.searchsorted(zone_start_ts, side="left"))
            if sig_pos is not None and origin_pos < len(df_5m):
                rec["confirmation_bars_5m"] = sig_pos - origin_pos
                rec["confirmation_calendar_minutes"] = (sig_time - zone_start_ts) / pd.Timedelta(minutes=1)
        rows.append(rec)

    enriched = pd.DataFrame(rows).set_index("signal_time")
    enriched.to_csv(OUT_DIR / "setup_quality_enriched_trades.csv")

    winners = enriched[enriched["is_win"]]
    losers = enriched[~enriched["is_win"]]
    print(f"Enriched rows: {len(enriched)}  Winners matched: {len(winners)}  Losers matched: {len(losers)}")
    print(f"Winners missing zone_start_time: {winners['zone_start_time'].isna().sum()}")
    print(f"Winners missing fvg_size_over_atr: {winners['fvg_size_over_atr'].isna().sum() if 'fvg_size_over_atr' in winners else 'col missing'}")

    def describe(s: pd.Series) -> dict:
        s = s.dropna()
        if len(s) == 0:
            return {}
        return {
            "n": len(s), "mean": s.mean(), "median": s.median(), "std": s.std(),
            "q25": s.quantile(0.25), "q75": s.quantile(0.75), "min": s.min(), "max": s.max(),
        }

    summary = {
        "winners_target1_R": describe(winners["target1_R"]),
        "winners_target2_R": describe(winners["target2_R"]),
        "losers_target1_R": describe(losers["target1_R"]),
        "losers_target2_R": describe(losers["target2_R"]),
        "winners_fvg_size_over_atr": describe(winners["fvg_size_over_atr"]) if "fvg_size_over_atr" in winners else {},
        "losers_fvg_size_over_atr": describe(losers["fvg_size_over_atr"]) if "fvg_size_over_atr" in losers else {},
        "winners_ob_body_over_atr": describe(winners["ob_body_over_atr"]) if "ob_body_over_atr" in winners else {},
        "losers_ob_body_over_atr": describe(losers["ob_body_over_atr"]) if "ob_body_over_atr" in losers else {},
        "winners_confirmation_bars_5m": describe(winners["confirmation_bars_5m"]) if "confirmation_bars_5m" in winners else {},
        "losers_confirmation_bars_5m": describe(losers["confirmation_bars_5m"]) if "confirmation_bars_5m" in losers else {},
    }

    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv(OUT_DIR / "setup_quality_summary_stats.csv")
    print(summary_df.to_string())

    # ---- threshold sweeps: does a target1_R cutoff enrich for winners? ----
    print("\n--- target1_R threshold sweep (win rate among trades with target1_R > x) ---")
    for thresh in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        subset = enriched[enriched["target1_R"] > thresh]
        if len(subset) > 0:
            print(f"target1_R > {thresh}: n={len(subset)}, win_rate={subset['is_win'].mean():.1%}")

    print("\n--- fvg_size_over_atr threshold sweep ---")
    if "fvg_size_over_atr" in enriched:
        for thresh in [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
            subset = enriched[enriched["fvg_size_over_atr"] > thresh]
            if len(subset) > 0:
                print(f"fvg_size_over_atr > {thresh}: n={len(subset)}, win_rate={subset['is_win'].mean():.1%}")

    print("\n--- ob_body_over_atr threshold sweep ---")
    if "ob_body_over_atr" in enriched:
        for thresh in [0.1, 0.2, 0.3, 0.4]:
            subset = enriched[enriched["ob_body_over_atr"] <= thresh]
            if len(subset) > 0:
                print(f"ob_body_over_atr <= {thresh}: n={len(subset)}, win_rate={subset['is_win'].mean():.1%}")

    print("\n--- confirmation_bars_5m threshold sweep ---")
    if "confirmation_bars_5m" in enriched:
        for thresh in [3, 6, 9, 15, 30, 45]:
            subset = enriched[enriched["confirmation_bars_5m"] <= thresh]
            if len(subset) > 0:
                print(f"confirmation_bars_5m <= {thresh}: n={len(subset)}, win_rate={subset['is_win'].mean():.1%}")

    print("\n--- target1_R <= threshold sweep (opposite direction) ---")
    for thresh in [0.15, 0.25, 0.4, 0.5, 0.75, 1.0]:
        subset = enriched[enriched["target1_R"] <= thresh]
        if len(subset) > 0:
            print(f"target1_R <= {thresh}: n={len(subset)}, win_rate={subset['is_win'].mean():.1%}")

    # ---- quartile-bucket win rate across ALL 151 trades, per dimension ----
    print("\n--- quartile-bucket win rates across all 151 trades ---")
    quartile_summary_rows = []
    for col in ["target1_R", "fvg_size_over_atr", "ob_body_over_atr", "confirmation_bars_5m"]:
        try:
            q = pd.qcut(enriched[col], 4, duplicates="drop")
        except ValueError:
            continue
        grp = enriched.groupby(q, observed=True)["is_win"].agg(["mean", "count"])
        print(f"\n{col}:")
        print(grp.to_string())
        for interval, r in grp.iterrows():
            quartile_summary_rows.append({"metric": col, "bucket": str(interval), "win_rate": r["mean"], "n": r["count"]})

    pd.DataFrame(quartile_summary_rows).to_csv(OUT_DIR / "setup_quality_quartile_win_rates.csv", index=False)


if __name__ == "__main__":
    main()
