"""Daily-only external/internal design vs. existing Weekly/Daily design.

RESEARCH ONLY. Reuses smc_validator primitives verbatim (no reimplementation
of pivot / BOS / CHoCH detection).

Fill/exit conventions are copied exactly from stock_hh_ll_tool/entry_exit.py:
  - signal on bar t -> fill at OPEN of bar t+1 (never signal-bar close)
  - stop = swing low active at signal, minus stop_atr_buffer_fraction * ATR14
  - stop fill = min(stop, that day's open) i.e. gap-down fills worse
  - trend exit at that day's close
Structures are computed once over the full series; this is identical to
recomputing on each prefix because detect_fractal_pivots only exposes a pivot
at bar confirmed_at = i + right, and compute_structure is a causal forward
state machine that consumes it there. (Verified explicitly in verify.py.)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/selvaganapathypari/Documents/trading_tool")
sys.path.insert(0, str(ROOT))

from smc_validator.data_ingestion.resample import align_htf_to_ltf, resample_ohlc  # noqa: E402
from smc_validator.patterns.order_blocks import atr  # noqa: E402
from smc_validator.structure.bos_choch import compute_structure  # noqa: E402
from smc_validator.structure.swings import detect_fractal_pivots  # noqa: E402

DATA = ROOT / "investigations" / "data"
ATR_BUF = 0.1


def structure_of(df, left, right):
    return compute_structure(df, detect_fractal_pivots(df, left, right), close_only=True)


def simulate(daily, trend_series, trigger_struct, require_arm, rearm_after_exit):
    """Generic long-only state machine.

    trend_series   : per-daily-bar 'bullish'/'bearish'/None regime (external
                     daily structure, or weekly structure aligned to daily)
    trigger_struct : the fast structure whose bullish CHoCH is the entry
                     trigger and whose swing_low sets the stop
    require_arm    : if True, an explicit bearish trigger-tier event must be
                     seen while the regime is bullish before an entry is
                     allowed (the literal "wait for a retracement" step)
    rearm_after_exit: if True, a fresh bearish trigger event is needed after
                     an exit before re-entering the same bullish regime
    """
    o = daily["open"].to_numpy()
    lo = daily["low"].to_numpy()
    cl = daily["close"].to_numpy()
    idx = daily.index
    a = atr(daily).to_numpy()

    ev = trigger_struct["event"].to_numpy()
    evd = trigger_struct["event_direction"].to_numpy()
    swl = trigger_struct["swing_low"].to_numpy(dtype=float)
    tr = np.asarray(trend_series, dtype=object)

    state = "FLAT"
    armed = False
    pend_stop = None
    entry = entry_i = stop = None
    trades = []

    for i in range(len(daily)):
        if state == "PENDING":
            entry, entry_i, stop, state = o[i], i, pend_stop, "LONG"
            continue

        if state == "LONG":
            if lo[i] <= stop:
                px = o[i] if o[i] < stop else stop
                reason = "stop"
            elif tr[i] == "bearish":
                px, reason = cl[i], "trend"
            else:
                continue
            risk = entry - stop
            trades.append(
                dict(
                    entry_date=idx[entry_i], exit_date=idx[i], entry=entry, exit=px, stop=stop,
                    total_r=(px - entry) / risk if risk > 0 else 0.0,
                    ret_pct=(px / entry - 1) * 100, hold_days=(idx[i] - idx[entry_i]).days,
                    hold_bars=i - entry_i, reason=reason, signal_time=idx[entry_i],
                )
            )
            state, entry, stop = "FLAT", None, None
            armed = not rearm_after_exit
            continue

        # FLAT
        if tr[i] != "bullish":
            armed = False
            continue
        if ev[i] in ("BOS", "CHoCH") and evd[i] == "bearish":
            armed = True
            continue
        entry_ok = ev[i] == "CHoCH" and evd[i] == "bullish"
        if entry_ok and (armed or not require_arm) and np.isfinite(swl[i]):
            buf = ATR_BUF * a[i] if np.isfinite(a[i]) else 0.0
            s = float(swl[i]) - buf
            if s < cl[i]:
                pend_stop, state, armed = s, "PENDING", False

    return pd.DataFrame(trades)


def run(variant, daily):
    ext = structure_of(daily, 5, 5)
    if variant["kind"] == "weekly":
        wk = resample_ohlc(daily, "W")
        if len(wk) < 30:
            return pd.DataFrame()
        wtr = align_htf_to_ltf(structure_of(wk, 5, 5)["trend"], daily.index)
        trend = wtr.to_numpy()
        trig = structure_of(daily, *variant["internal"])
    else:
        trend = ext["trend"].to_numpy()
        trig = structure_of(daily, *variant["internal"])
    return simulate(daily, trend, trig, variant["require_arm"], variant["rearm"])


VARIANTS = {
    "WEEKLY/DAILY 5-5 wk + 3-3 d (baseline)": dict(kind="weekly", internal=(3, 3), require_arm=False, rearm=False),
    "DAILY-ONLY ext5-5 + int2-2": dict(kind="daily", internal=(2, 2), require_arm=True, rearm=True),
    "DAILY-ONLY ext5-5 + int3-3": dict(kind="daily", internal=(3, 3), require_arm=True, rearm=True),
    "DAILY-ONLY ext5-5 + int1-1": dict(kind="daily", internal=(1, 1), require_arm=True, rearm=True),
    "DAILY-ONLY ext5-5 + int3-3 (no explicit arm)": dict(kind="daily", internal=(3, 3), require_arm=False, rearm=False),
    "DAILY-ONLY ext5-5 + int2-2 (no explicit arm)": dict(kind="daily", internal=(2, 2), require_arm=False, rearm=False),
    "DAILY-ONLY ext8-8 + int3-3": dict(kind="daily", internal=(3, 3), require_arm=True, rearm=True),
    "HYBRID weekly-trend + daily ext5-5 agree, int3-3 trigger": dict(kind="hybrid", internal=(3, 3), require_arm=True, rearm=True),
}


def run_variant(name, v, daily):
    if v["kind"] == "hybrid":
        ext = structure_of(daily, 5, 5)
        wk = resample_ohlc(daily, "W")
        wtr = align_htf_to_ltf(structure_of(wk, 5, 5)["trend"], daily.index).to_numpy()
        e = ext["trend"].to_numpy()
        trend = np.array(
            ["bullish" if (a == "bullish" and b == "bullish") else ("bearish" if (a == "bearish" or b == "bearish") else None)
             for a, b in zip(wtr, e)], dtype=object)
        trig = structure_of(daily, *v["internal"])
        return simulate(daily, trend, trig, v["require_arm"], v["rearm"])
    if name.startswith("DAILY-ONLY ext8-8"):
        ext = structure_of(daily, 8, 8)
        trig = structure_of(daily, *v["internal"])
        return simulate(daily, ext["trend"].to_numpy(), trig, v["require_arm"], v["rearm"])
    return run(v, daily)


def summarize(t):
    if len(t) == 0:
        return dict(n=0)
    w = t["total_r"] > 0
    gw = t.loc[w, "total_r"].sum()
    gl = -t.loc[~w, "total_r"].sum()
    span = (t["signal_time"].max() - t["signal_time"].min()).days
    return dict(
        n=len(t), win_rate=float(w.mean()), expectancy_r=float(t["total_r"].mean()),
        median_r=float(t["total_r"].median()),
        avg_win_r=float(t.loc[w, "total_r"].mean()) if w.any() else 0.0,
        avg_loss_r=float(t.loc[~w, "total_r"].mean()) if (~w).any() else 0.0,
        profit_factor=float(gw / gl) if gl > 0 else float("inf"),
        avg_hold=float(t["hold_days"].mean()), med_hold=float(t["hold_days"].median()),
        pct_hold_lt5=float((t["hold_days"] < 5).mean()), pct_hold_gt60=float((t["hold_days"] > 60).mean()),
        mean_ret_pct=float(t["ret_pct"].mean()), median_ret_pct=float(t["ret_pct"].median()),
        worst_pct=float(t["ret_pct"].min()),
        stop_exit_frac=float((t["reason"] == "stop").mean()),
        trades_per_month_per_stock=len(t) / (span / 30.0) / 35 if span > 0 else float("nan"),
    )


if __name__ == "__main__":
    files = sorted(DATA.glob("*.parquet"))
    all_res = {k: [] for k in VARIANTS}
    for f in files:
        d = pd.read_parquet(f)
        d = d.dropna().sort_index()
        for name, v in VARIANTS.items():
            t = run_variant(name, v, d)
            if len(t):
                t = t.copy()
                t["symbol"] = f.stem
                all_res[name].append(t)

    rows = []
    for name in VARIANTS:
        pooled = pd.concat(all_res[name]) if all_res[name] else pd.DataFrame()
        s = summarize(pooled)
        s["variant"] = name
        rows.append(s)
        pooled.to_csv(Path(__file__).parent / f"trades_{abs(hash(name))%10**8}.csv", index=False)
    out = pd.DataFrame(rows).set_index("variant")
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print(out.round(3).to_string())
    out.to_csv(Path(__file__).parent / "summary.csv")
