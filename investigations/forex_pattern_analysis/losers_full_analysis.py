"""Characterize the LOSING trades (total_r <= 0) from the validated SMC
EURUSD backtest, across the same three angles the winners-side agents cover:
  A) timing / session / volatility
  B) setup quality at entry (R:R, FVG size, OB body size, zone-to-confirmation lag)
  C) exit / management characteristics (exit_reason mix, leg1 vs leg2 contribution,
     holding duration, direction split)

Read-only reuse of scripts/run_backtest.py::build_signals and
smc_validator/*::simulate_trades / detect_order_blocks / detect_fvg / atr /
resample_ohlc. Nothing here modifies smc_validator/ or scripts/ — only
imports from them. This script is fully self-contained (does not depend on
any other agent's output files, since those run in parallel and may be
overwritten) — it regenerates the 151-trade sample itself.

Bucket definitions are deliberately identical to what a winners-side session/
ATR analysis would plausibly use:
  - sessions: Asia 00:00-09:00 UTC, London 08:00-17:00 UTC, NY 13:00-22:00 UTC
    (with London/NY and Asia/London overlaps called out separately)
  - ATR terciles: trailing 60-calendar-day window of 15m ATR(14), low/med/high
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from smc_validator.backtest.engine import simulate_trades
from smc_validator.config import load_config
from smc_validator.data_ingestion.resample import resample_ohlc, align_htf_to_ltf
from smc_validator.patterns.order_blocks import atr as compute_atr, detect_order_blocks
from smc_validator.patterns.fvg import detect_fvg
from smc_validator.structure.bos_choch import compute_structure
from smc_validator.structure.swings import detect_fractal_pivots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from run_backtest import build_signals  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 140)


def classify_session(ts: pd.Timestamp) -> str:
    h = ts.hour
    asia = 0 <= h < 9
    london = 8 <= h < 17
    ny = 13 <= h < 22
    if london and ny:
        return "London/NY overlap"
    if asia and london:
        return "Asia/London overlap"
    if london:
        return "London"
    if ny:
        return "New York"
    if asia:
        return "Asia"
    return "Off-session (22:00-00:00 UTC gap)"


def tercile(p):
    if pd.isna(p):
        return "insufficient_history"
    if p < 1 / 3:
        return "low"
    if p < 2 / 3:
        return "medium"
    return "high"


def main() -> None:
    cfg = load_config()
    print("Building signals (full pipeline)...")
    bars, signals = build_signals(cfg)
    print(f"{len(signals)} candidate signals from {len(bars)} 5m bars.")

    trades = simulate_trades(bars, signals, staged_exit_split=tuple(cfg["risk"]["staged_exit_split"]))
    print(f"{len(trades)} simulated trades.")
    trades = trades.copy()
    trades["is_win"] = trades["total_r"] > 0
    n_all = len(trades)
    n_win = int(trades["is_win"].sum())
    n_loss = n_all - n_win
    print(f"Win rate: {n_win/n_all:.3%} ({n_win}/{n_all}); losers: {n_loss}")

    # ============ shared enrichment (also needed for A) ============
    trades["session"] = trades["signal_time"].apply(classify_session)
    trades["dow"] = trades["signal_time"].dt.day_name()
    trades["hour_utc"] = trades["signal_time"].dt.hour
    trades["month"] = trades["signal_time"].dt.to_period("M").astype(str)
    trades["quarter"] = trades["signal_time"].dt.to_period("Q").astype(str)

    df_15m = resample_ohlc(bars, "15min")
    atr_15m = compute_atr(df_15m, window=14)
    atr_df = atr_15m.rename("atr15").reset_index()
    atr_df.columns = ["timestamp", "atr15"]
    atr_df = atr_df.dropna(subset=["atr15"]).sort_values("timestamp")
    atr_series = atr_df.set_index("timestamp")["atr15"]

    trades_sorted = trades.sort_values("signal_time")
    merged = pd.merge_asof(trades_sorted, atr_df, left_on="signal_time", right_on="timestamp", direction="backward")
    merged = merged.drop(columns=["timestamp"])

    def trailing_percentile(row):
        window_start = row["signal_time"] - pd.Timedelta(days=60)
        hist = atr_series.loc[window_start:row["signal_time"]].dropna()
        if len(hist) < 20:
            return np.nan
        return (hist < row["atr15"]).mean()

    merged["atr_trailing_pct"] = merged.apply(trailing_percentile, axis=1)
    merged["vol_regime"] = merged["atr_trailing_pct"].apply(tercile)

    # ============ B) setup quality — re-derive OB/FVG/zone bookkeeping ============
    fvg_15m = detect_fvg(df_15m, min_gap_size=cfg["fair_value_gaps"]["min_gap_size_atr_fraction"] * atr_15m.fillna(0))
    structure_15m = compute_structure(
        df_15m,
        detect_fractal_pivots(df_15m, cfg["structure"]["external"]["left_bars"], cfg["structure"]["external"]["right_bars"]),
        close_only=cfg["structure"]["close_only_break"],
    )
    ob_15m = detect_order_blocks(
        df_15m,
        atr_15m,
        balance_body_atr_fraction=cfg["order_blocks"]["balance_candle_body_atr_fraction"],
        expansion_atr_multiple=cfg["order_blocks"]["expansion_move_atr_multiple"],
        lookahead=5,
        fvg_flags=fvg_15m if cfg["order_blocks"]["require_associated_fvg"] else None,
        structure_event=structure_15m["event"],
        structure_event_direction=structure_15m["event_direction"],
    )

    body_15m = (df_15m["close"] - df_15m["open"]).abs()
    ob_body_atr_ratio_15m = body_15m / atr_15m  # ratio at the OB bar itself

    zone_start_15m = ob_15m["bullish_ob"] | ob_15m["bearish_ob"]
    lookahead = 5

    # for each 15m bar index i where zone_start, find first matching-direction
    # FVG bar in [i, i+lookahead-1] and compute its gap size / ATR at bar i
    fvg_size_atr_ratio = pd.Series(np.nan, index=df_15m.index)
    n15 = len(df_15m)
    zone_start_idx = np.flatnonzero(zone_start_15m.to_numpy())
    is_bullish_arr = ob_15m["bullish_ob"].to_numpy()
    fvg_top = fvg_15m["fvg_top"].to_numpy()
    fvg_bottom = fvg_15m["fvg_bottom"].to_numpy()
    fvg_bull = fvg_15m["fvg_bullish"].to_numpy()
    fvg_bear = fvg_15m["fvg_bearish"].to_numpy()
    atr_arr = atr_15m.to_numpy()
    fvg_ratio_arr = fvg_size_atr_ratio.to_numpy().copy()
    for i in zone_start_idx:
        want_bull = is_bullish_arr[i]
        for j in range(i, min(i + lookahead, n15)):
            match = fvg_bull[j] if want_bull else fvg_bear[j]
            if match:
                gap = fvg_top[j] - fvg_bottom[j]
                if atr_arr[i] and not np.isnan(atr_arr[i]):
                    fvg_ratio_arr[i] = gap / atr_arr[i]
                break
    fvg_size_atr_ratio = pd.Series(fvg_ratio_arr, index=df_15m.index)

    # zone bookkeeping identical to build_signals' loop, but also track the
    # anchor OB bar's timestamp / body-atr-ratio / fvg-size-atr-ratio so we
    # can align them onto 5m and attach to each signal
    timeout = cfg["ltf_confirmation"]["timeout_bars"]
    zone_active_15m = pd.Series(False, index=df_15m.index)
    zone_anchor_ts_15m = pd.Series(pd.NaT, index=df_15m.index, dtype="datetime64[ns, UTC]")
    zone_anchor_ob_body_ratio_15m = pd.Series(np.nan, index=df_15m.index)
    zone_anchor_fvg_ratio_15m = pd.Series(np.nan, index=df_15m.index)
    bars_since_zone_start = None
    cur_anchor_ts = None
    cur_ob_body_ratio = None
    cur_fvg_ratio = None
    for i in range(len(df_15m)):
        if zone_start_15m.iloc[i]:
            bars_since_zone_start = 0
            cur_anchor_ts = df_15m.index[i]
            cur_ob_body_ratio = ob_body_atr_ratio_15m.iloc[i]
            cur_fvg_ratio = fvg_size_atr_ratio.iloc[i]
        elif bars_since_zone_start is not None:
            bars_since_zone_start += 1
        if bars_since_zone_start is not None and bars_since_zone_start <= timeout:
            zone_active_15m.iloc[i] = True
            zone_anchor_ts_15m.iloc[i] = cur_anchor_ts
            zone_anchor_ob_body_ratio_15m.iloc[i] = cur_ob_body_ratio
            zone_anchor_fvg_ratio_15m.iloc[i] = cur_fvg_ratio

    zone_anchor_ts_5m = align_htf_to_ltf(zone_anchor_ts_15m, bars.index)
    zone_anchor_ob_body_ratio_5m = align_htf_to_ltf(zone_anchor_ob_body_ratio_15m, bars.index)
    zone_anchor_fvg_ratio_5m = align_htf_to_ltf(zone_anchor_fvg_ratio_15m, bars.index)

    # attach setup-quality fields straight from `signals` (indexed by signal_time)
    sig = signals.copy()
    sig["target1_R"] = (sig["target1"] - sig["entry"]).abs() / (sig["entry"] - sig["stop"]).abs()
    sig["target2_R"] = (sig["target2"] - sig["entry"]).abs() / (sig["entry"] - sig["stop"]).abs()
    sig["zone_anchor_ts"] = zone_anchor_ts_5m.reindex(sig.index)
    sig["ob_body_atr_ratio"] = zone_anchor_ob_body_ratio_5m.reindex(sig.index)
    sig["fvg_size_atr_ratio"] = zone_anchor_fvg_ratio_5m.reindex(sig.index)
    sig["bars_zone_to_confirmation_5m"] = (
        (sig.index - sig["zone_anchor_ts"]) / pd.Timedelta(minutes=5)
    )

    setup_cols = ["target1_R", "target2_R", "ob_body_atr_ratio", "fvg_size_atr_ratio", "bars_zone_to_confirmation_5m"]
    sig_reset = sig[setup_cols].reset_index()
    sig_reset.columns = ["signal_time"] + setup_cols
    merged = merged.merge(sig_reset, on="signal_time", how="left")

    # ============ C) exit/management — instrument a local copy of the sim loop ============
    def simulate_with_timestamps(bars, signals):
        bar_index = bars.index
        positions = bar_index.get_indexer(signals.index)
        out = []
        for sig_idx, pos in zip(signals.index, positions):
            if pos == -1 or pos + 1 >= len(bars):
                continue
            row = signals.loc[sig_idx]
            direction, entry, stop, t1, t2 = row["direction"], row["entry"], row["stop"], row["target1"], row["target2"]
            risk = abs(entry - stop)
            if risk == 0:
                continue
            leg1_done = False
            t1_time = None
            exit_time = None
            exit_reason = None
            for j in range(pos + 1, len(bars)):
                bar = bars.iloc[j]
                if direction == "bullish":
                    hit_stop, hit_t1, hit_t2 = bar["low"] <= stop, bar["high"] >= t1, bar["high"] >= t2
                else:
                    hit_stop, hit_t1, hit_t2 = bar["high"] >= stop, bar["low"] <= t1, bar["low"] <= t2
                if not leg1_done:
                    if hit_stop:
                        exit_time = bars.index[j]
                        exit_reason = "stop_before_target1"
                        break
                    if hit_t1:
                        leg1_done = True
                        t1_time = bars.index[j]
                        continue
                else:
                    if hit_stop:
                        exit_time = bars.index[j]
                        exit_reason = "stop_after_partial"
                        break
                    if hit_t2:
                        exit_time = bars.index[j]
                        exit_reason = "both_targets"
                        break
            else:
                exit_time = bars.index[-1]
                exit_reason = "unresolved_end_of_data"
            out.append({"signal_time": sig_idx, "t1_time": t1_time, "exit_time": exit_time})
        return pd.DataFrame(out)

    timing_df = simulate_with_timestamps(bars, signals)
    merged = merged.merge(timing_df, on="signal_time", how="left")
    merged["bars_to_resolution_5m"] = (merged["exit_time"] - merged["signal_time"]) / pd.Timedelta(minutes=5)
    merged["bars_to_t1_5m"] = (merged["t1_time"] - merged["signal_time"]) / pd.Timedelta(minutes=5)

    merged.to_csv(OUT_DIR / "losers_trades_enriched.csv", index=False)

    losers = merged[~merged["is_win"]].copy()
    winners_ref = merged[merged["is_win"]].copy()
    losers.to_csv(OUT_DIR / "losers_full_enriched.csv", index=False)
    n_l = len(losers)
    print(f"\n=== LOSERS: {n_l} / {n_all} ({n_l/n_all:.1%}) ===\n")

    # ================= A) TIMING / SESSION / VOLATILITY =================
    print("\n########## A) TIMING / SESSION / VOLATILITY (losers) ##########")
    print("\n--- Session ---")
    for s in sorted(merged["session"].unique()):
        all_n = (merged["session"] == s).sum()
        loss_n = (losers["session"] == s).sum()
        print(f"  {s:28s}: {loss_n:3d} losers ({loss_n/n_l:5.1%} of losers) | loss rate within session: {loss_n/all_n:.1%} ({loss_n}/{all_n})")

    print("\n--- Day of week ---")
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for d in dow_order:
        all_n = (merged["dow"] == d).sum()
        if all_n == 0:
            continue
        loss_n = (losers["dow"] == d).sum()
        print(f"  {d:10s}: {loss_n:3d} losers ({loss_n/n_l:5.1%} of losers) | loss rate: {loss_n/all_n:.1%} ({loss_n}/{all_n})")

    print("\n--- Hour of day (UTC) ---")
    for h in sorted(merged["hour_utc"].unique()):
        all_n = (merged["hour_utc"] == h).sum()
        loss_n = (losers["hour_utc"] == h).sum()
        print(f"  {h:02d}:00 UTC: {loss_n:3d} losers ({loss_n/n_l:5.1%} of losers) | loss rate: {loss_n/all_n:.1%} ({loss_n}/{all_n})")

    print("\n--- Month ---")
    for m in sorted(merged["month"].unique()):
        all_n = (merged["month"] == m).sum()
        loss_n = (losers["month"] == m).sum()
        print(f"  {m}: {loss_n:3d} losers ({loss_n/n_l:5.1%} of losers) | loss rate: {loss_n/all_n:.1%} ({loss_n}/{all_n})")

    print("\n--- Quarter ---")
    for q in sorted(merged["quarter"].unique()):
        all_n = (merged["quarter"] == q).sum()
        loss_n = (losers["quarter"] == q).sum()
        print(f"  {q}: {loss_n:3d} losers ({loss_n/n_l:5.1%} of losers) | loss rate: {loss_n/all_n:.1%} ({loss_n}/{all_n})")

    print("\n--- Volatility regime (trailing 60d ATR tercile) ---")
    vol_order = ["low", "medium", "high", "insufficient_history"]
    for v in vol_order:
        all_n = (merged["vol_regime"] == v).sum()
        if all_n == 0:
            continue
        loss_n = (losers["vol_regime"] == v).sum()
        print(f"  {v:22s}: {loss_n:3d} losers ({loss_n/n_l:5.1%} of losers) | loss rate: {loss_n/all_n:.1%} ({loss_n}/{all_n})")

    print(f"\n  ATR15 (raw, price units) — losers mean {losers['atr15'].mean():.6f} median {losers['atr15'].median():.6f}")
    print(f"  ATR15 (raw, price units) — winners mean {winners_ref['atr15'].mean():.6f} median {winners_ref['atr15'].median():.6f}")
    print(f"  ATR15 (raw, price units) — all mean {merged['atr15'].mean():.6f} median {merged['atr15'].median():.6f}")

    # ================= B) SETUP QUALITY =================
    print("\n\n########## B) SETUP QUALITY AT ENTRY (losers) ##########")

    def q(s):
        s = s.dropna()
        return s.quantile([0.25, 0.5, 0.75])

    for col, label in [("target1_R", "target1_R (RR to target1)"), ("target2_R", "target2_R (RR to target2)")]:
        print(f"\n--- {label} ---")
        for name, d in [("losers", losers), ("winners", winners_ref), ("all", merged)]:
            v = d[col].dropna()
            print(f"  {name:8s}: mean={v.mean():.2f} median={v.median():.2f} Q1={v.quantile(.25):.2f} Q3={v.quantile(.75):.2f} n={len(v)}")

    print("\n--- FVG size / ATR15 at anchor OB bar ---")
    for name, d in [("losers", losers), ("winners", winners_ref), ("all", merged)]:
        v = d["fvg_size_atr_ratio"].dropna()
        print(f"  {name:8s}: mean={v.mean():.3f} median={v.median():.3f} Q1={v.quantile(.25):.3f} Q3={v.quantile(.75):.3f} n={len(v)}")

    print("\n--- OB candle body / ATR15 at anchor OB bar ---")
    for name, d in [("losers", losers), ("winners", winners_ref), ("all", merged)]:
        v = d["ob_body_atr_ratio"].dropna()
        print(f"  {name:8s}: mean={v.mean():.3f} median={v.median():.3f} Q1={v.quantile(.25):.3f} Q3={v.quantile(.75):.3f} n={len(v)}")

    print("\n--- Bars (5m) elapsed: zone becoming active -> LTF confirmation firing ---")
    for name, d in [("losers", losers), ("winners", winners_ref), ("all", merged)]:
        v = d["bars_zone_to_confirmation_5m"].dropna()
        print(f"  {name:8s}: mean={v.mean():.1f} median={v.median():.1f} Q1={v.quantile(.25):.1f} Q3={v.quantile(.75):.1f} n={len(v)}")

    # ================= C) EXIT / MANAGEMENT =================
    print("\n\n########## C) EXIT / MANAGEMENT CHARACTERISTICS (losers) ##########")
    print("\n--- exit_reason distribution ---")
    for name, d in [("losers", losers), ("winners", winners_ref), ("all", merged)]:
        n = len(d)
        vc = d["exit_reason"].value_counts()
        print(f"  {name}:")
        for reason, c in vc.items():
            print(f"    {reason:24s}: {c:3d} ({c/n:5.1%})")

    stop_after_partial = losers[losers["exit_reason"] == "stop_after_partial"]
    print(f"\n--- stop_after_partial losers: {len(stop_after_partial)}/{n_l} ({len(stop_after_partial)/n_l:.1%}) ---")
    if len(stop_after_partial) > 0:
        leg1_contrib = 0.5 * stop_after_partial["leg1_r"]
        leg2_contrib = 0.5 * stop_after_partial["leg2_r"]
        print(f"  mean leg1 R contribution (0.5*leg1_r): {leg1_contrib.mean():.3f}")
        print(f"  mean leg2 R contribution (0.5*leg2_r): {leg2_contrib.mean():.3f}  (fixed at -0.5 by construction)")
        print(f"  mean total_r for this subset: {stop_after_partial['total_r'].mean():.3f}")
        print(f"  median leg1_r (raw, before 0.5 split): {stop_after_partial['leg1_r'].median():.3f}")

    stop_before_t1 = losers[losers["exit_reason"] == "stop_before_target1"]
    print(f"\n--- stop_before_target1 losers: {len(stop_before_t1)}/{n_l} ({len(stop_before_t1)/n_l:.1%}) — total_r fixed at -1.0 ---")

    print("\n--- Holding duration (5m bars) until final resolution ---")
    for name, d in [("losers", losers), ("winners", winners_ref), ("all", merged)]:
        v = d["bars_to_resolution_5m"].dropna()
        hrs = v * 5 / 60
        print(f"  {name:8s}: mean={v.mean():.1f} bars ({hrs.mean():.1f}h) median={v.median():.1f} bars ({v.median()*5/60:.1f}h) Q1={v.quantile(.25):.1f} Q3={v.quantile(.75):.1f} n={len(v)}")

    print("\n--- Bars (5m) to target1 (where reached, incl. stop_after_partial and both_targets) ---")
    for name, d in [("losers", losers), ("winners", winners_ref), ("all", merged)]:
        v = d["bars_to_t1_5m"].dropna()
        if len(v) == 0:
            print(f"  {name}: n=0")
            continue
        print(f"  {name:8s}: mean={v.mean():.1f} median={v.median():.1f} n={len(v)}")

    print("\n--- Direction split ---")
    for name, d in [("losers", losers), ("winners", winners_ref), ("all", merged)]:
        n = len(d)
        vc = d["direction"].value_counts()
        parts = ", ".join(f"{k}={c} ({c/n:.1%})" for k, c in vc.items())
        print(f"  {name:8s}: {parts}")

    print("\nDone.")


if __name__ == "__main__":
    main()
