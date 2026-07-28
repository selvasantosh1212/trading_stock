"""Does a genuine LIQUIDITY-SWEEP exit beat both the no-target baseline and a
hard price target?

Concept (smc_validator/liquidity/sweep_detection.py): price runs into a real
liquidity pool (a major untaken prior high), takes it out intraday, then CLOSES
back below it -> the pool has been consumed and rejected -> exit. This is a
TRAILING/event exit, not a price cap, so in principle it should not truncate the
big trends the way a fixed T2 does.

Also runs a paired bootstrap on the expectancy differences so the comparison
isn't read off point estimates alone.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from target_lab import (HTF_L, HTF_R, LTF_L, LTF_R, STOP_ATR_FRAC, build_level_pool,  # noqa: E402
                        load_all, pick_targets, summarize, untaken_levels_at)
from smc_validator.data_ingestion.resample import align_htf_to_ltf, resample_ohlc  # noqa: E402
from smc_validator.patterns.order_blocks import atr  # noqa: E402
from smc_validator.structure.bos_choch import compute_structure  # noqa: E402
from smc_validator.structure.swings import detect_fractal_pivots  # noqa: E402

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)

MAJOR = {"weekly_swing", "prev_month"}


def run(sym, daily, t1_min_r=1.5):
    weekly = resample_ohlc(daily, "W")
    htf = compute_structure(weekly, detect_fractal_pivots(weekly, HTF_L, HTF_R), close_only=True)
    ltf = compute_structure(daily, detect_fractal_pivots(daily, LTF_L, LTF_R), close_only=True)
    trend = align_htf_to_ltf(htf["trend"], daily.index).to_numpy()
    n = len(daily)
    idx = daily.index
    op, hi, lo, cl = (daily[c].to_numpy() for c in ("open", "high", "low", "close"))
    ev, evd = ltf["event"].to_numpy(), ltf["event_direction"].to_numpy()
    swl = ltf["swing_low"].to_numpy()
    a = atr(daily).to_numpy()
    levels = build_level_pool(daily, weekly, "long")

    rows, i = [], 0
    while i < n - 2:
        if not (trend[i] == "bullish" and ev[i] == "CHoCH" and evd[i] == "bullish") or pd.isna(swl[i]):
            i += 1
            continue
        ei = i + 1
        entry = float(op[ei])
        stop0 = float(swl[i]) - (STOP_ATR_FRAC * a[i] if not np.isnan(a[i]) else 0.0)
        risk = entry - stop0
        if risk <= 0:
            i = ei + 1
            continue
        pool = untaken_levels_at(levels, i, hi, lo, "long", entry)
        t1, _k, _t2, _k2 = pick_targets(pool, entry, risk, "long", t1_min_r, t1_min_r * 2)

        out = {}
        for mode in ("baseline", "sweep_full", "t1_then_sweep"):
            stop, hit_t1 = stop0, False
            leg1 = leg2 = None
            reason, ebar = None, n - 1
            j = ei + 1
            done = False
            while j < n:
                if lo[j] <= stop:
                    fill = op[j] if op[j] < stop else stop
                    r = (fill - entry) / risk
                    if hit_t1:
                        leg2, reason = r, "stop_after_t1"
                    else:
                        leg1 = leg2 = r
                        reason = "stop"
                    ebar, done = j, True
                    break
                if mode == "t1_then_sweep" and not hit_t1 and t1 is not None and hi[j] >= t1:
                    hit_t1, leg1 = True, (t1 - entry) / risk
                    j += 1
                    continue
                if mode in ("sweep_full", "t1_then_sweep"):
                    # sweep-and-reject of a MAJOR untaken pool, knowable at j's close
                    swept = False
                    for price, kind in untaken_levels_at(levels, j - 1, hi, lo, "long", entry):
                        if kind in MAJOR and hi[j] > price and cl[j] < price:
                            swept = True
                            break
                    if swept and (mode == "sweep_full" or hit_t1):
                        if j + 1 >= n:
                            break
                        r = (op[j + 1] - entry) / risk
                        if hit_t1:
                            leg2 = r
                        else:
                            leg1 = leg2 = r
                        reason, ebar, done = "sweep_reject", j + 1, True
                        break
                if trend[j] == "bearish":
                    if j + 1 >= n:
                        break
                    r = (op[j + 1] - entry) / risk
                    if hit_t1:
                        leg2 = r
                    else:
                        leg1 = leg2 = r
                    reason, ebar, done = "trend_exit", j + 1, True
                    break
                j += 1
            if not done:
                r = (cl[n - 1] - entry) / risk
                if hit_t1:
                    leg2 = r
                else:
                    leg1 = leg2 = r
                reason, ebar = "open_at_end", n - 1
            if leg2 is None:
                leg2 = leg1
            out[mode] = (0.5 * leg1 + 0.5 * leg2 if hit_t1 else leg1, reason, ebar, hit_t1)

        rows.append({"symbol": sym, "signal_date": idx[i], "entry": entry, "risk_pct": risk / entry * 100,
                     "t1": t1,
                     **{f"{m}_r": out[m][0] for m in out},
                     **{f"{m}_reason": out[m][1] for m in out},
                     "t1_hit": out["t1_then_sweep"][3]})
        i = max(out["baseline"][2], ei) + 1
    return pd.DataFrame(rows)


data = load_all()
frames = [run(s, d) for s, d in data.items()]
tr = pd.concat([f for f in frames if len(f)], ignore_index=True)
tr.to_csv(Path(__file__).parent / "trades_sweep.csv", index=False)

print(f"n = {len(tr)} trades\n")
for m in ("baseline", "sweep_full", "t1_then_sweep"):
    s = summarize(tr, f"{m}_r")
    print(f"{m:15s} n={s['n']:4d} win={s['win_rate']:5.1f}% exp={s['expectancy_r']:+.3f}R "
          f"med={s['median_r']:+.3f}R PF={s['profit_factor']:.2f} maxDD={s['max_dd_r']:.2f}R "
          f"best={s['best_r']:.1f}R")
print()
for m in ("baseline", "sweep_full", "t1_then_sweep"):
    print(m, dict(tr[f"{m}_reason"].value_counts()))

print("\n### paired bootstrap on expectancy difference vs baseline (10k resamples) ###")
rng = np.random.default_rng(7)
base = tr["baseline_r"].to_numpy()
for m in ("sweep_full", "t1_then_sweep"):
    d = tr[f"{m}_r"].to_numpy() - base
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
    print(f"  {m:15s} mean diff {d.mean():+.3f}R   95% CI [{np.percentile(boot,2.5):+.3f}, "
          f"{np.percentile(boot,97.5):+.3f}]   P(diff>0)={np.mean(boot>0)*100:.1f}%")

# also bootstrap the main t1run design from run_main
main = pd.read_csv(Path(__file__).parent / "trades_long_main.csv")
print("\n### same bootstrap for the run_main variants (T1>=1.0R) ###")
b = main["baseline_r"].to_numpy()
for v in ("t1t2", "t1run", "t1run_be", "t1run_trail"):
    d = main[f"{v}_r"].to_numpy() - b
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
    print(f"  {v:13s} mean diff {d.mean():+.3f}R   95% CI [{np.percentile(boot,2.5):+.3f}, "
          f"{np.percentile(boot,97.5):+.3f}]   P(diff>0)={np.mean(boot>0)*100:.1f}%")
