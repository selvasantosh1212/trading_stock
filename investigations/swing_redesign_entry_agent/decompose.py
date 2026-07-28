"""Decompose ENTRY-rule effect from EXIT-rule effect.

The naive comparison changes both at once (daily-only design also exits on a
daily trend flip instead of a weekly one). This runs the 2x2.
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


def simulate(daily, entry_trend, exit_trend, trig, require_arm=True, rearm=True, max_stop_pct=None):
    o, lo, cl = daily["open"].to_numpy(), daily["low"].to_numpy(), daily["close"].to_numpy()
    idx, a = daily.index, atr(daily).to_numpy()
    ev, evd = trig["event"].to_numpy(), trig["event_direction"].to_numpy()
    swl = trig["swing_low"].to_numpy(dtype=float)
    etr, xtr = np.asarray(entry_trend, dtype=object), np.asarray(exit_trend, dtype=object)

    state, armed, pend, entry, entry_i, stop = "FLAT", False, None, None, None, None
    trades = []
    for i in range(len(daily)):
        if state == "PENDING":
            entry, entry_i, stop, state = o[i], i, pend, "LONG"
            continue
        if state == "LONG":
            if lo[i] <= stop:
                px, reason = (o[i] if o[i] < stop else stop), "stop"
            elif xtr[i] == "bearish":
                px, reason = cl[i], "trend"
            else:
                continue
            risk = entry - stop
            trades.append(dict(entry_date=idx[entry_i], exit_date=idx[i], entry=entry, exit=px,
                               total_r=(px - entry) / risk if risk > 0 else 0.0,
                               ret_pct=(px / entry - 1) * 100, hold_days=(idx[i] - idx[entry_i]).days,
                               reason=reason, signal_time=idx[entry_i],
                               stop_pct=(entry - stop) / entry * 100))
            state, armed = "FLAT", (not rearm)
            continue
        if etr[i] != "bullish":
            armed = False
            continue
        if ev[i] in ("BOS", "CHoCH") and evd[i] == "bearish":
            armed = True
            continue
        if ev[i] == "CHoCH" and evd[i] == "bullish" and (armed or not require_arm) and np.isfinite(swl[i]):
            buf = ATR_BUF * a[i] if np.isfinite(a[i]) else 0.0
            s = float(swl[i]) - buf
            if s < cl[i] and (max_stop_pct is None or (cl[i] - s) / cl[i] * 100 <= max_stop_pct):
                pend, state, armed = s, "PENDING", False
    return pd.DataFrame(trades)


def summarize(t):
    if len(t) == 0:
        return dict(n=0)
    w = t["total_r"] > 0
    gl = -t.loc[~w, "total_r"].sum()
    span = (t["signal_time"].max() - t["signal_time"].min()).days
    days_held = t["hold_days"].sum()
    return dict(n=len(t), win_rate=round(float(w.mean()), 3), exp_r=round(float(t["total_r"].mean()), 3),
                med_r=round(float(t["total_r"].median()), 3),
                pf=round(float(t.loc[w, "total_r"].sum() / gl), 2) if gl > 0 else np.inf,
                avg_hold=round(float(t["hold_days"].mean()), 1), med_hold=float(t["hold_days"].median()),
                p10_hold=float(t["hold_days"].quantile(.1)), p90_hold=float(t["hold_days"].quantile(.9)),
                mean_ret=round(float(t["ret_pct"].mean()), 2), med_ret=round(float(t["ret_pct"].median()), 2),
                worst=round(float(t["ret_pct"].min()), 1),
                R_per_100d=round(float(t["total_r"].sum()) / days_held * 100, 3),
                tr_per_mo_35=round(len(t) / (span / 30.0), 2) if span > 0 else np.nan,
                med_stop_pct=round(float(t["stop_pct"].median()), 2))


if __name__ == "__main__":
    files = sorted(DATA.glob("*.parquet"))
    combos = {}
    for f in files:
        d = pd.read_parquet(f).dropna().sort_index()
        ext5 = structure_of(d, 5, 5)
        int2, int3 = structure_of(d, 2, 2), structure_of(d, 3, 3)
        wk = resample_ohlc(d, "W")
        wtr = align_htf_to_ltf(structure_of(wk, 5, 5)["trend"], d.index).to_numpy()
        etr5 = ext5["trend"].to_numpy()

        specs = {
            "A entryWK / exitWK / trig d3-3 (EXISTING)": (wtr, wtr, int3, False, False, None),
            "B entryDAILYext / exitDAILYext / trig d2-2 (LITERAL)": (etr5, etr5, int2, True, True, None),
            "C entryDAILYext / exitDAILYext / trig d3-3": (etr5, etr5, int3, True, True, None),
            "D entryDAILYext / exitWK / trig d2-2": (etr5, wtr, int2, True, True, None),
            "E entryWK / exitDAILYext / trig d3-3": (wtr, etr5, int3, False, False, None),
            "F entryDAILYext+WK agree / exitDAILYext / trig d2-2": (
                np.array(["bullish" if (x == "bullish" and y == "bullish") else "other" for x, y in zip(wtr, etr5)], dtype=object),
                etr5, int2, True, True, None),
            "G LITERAL + stop cap 12%": (etr5, etr5, int2, True, True, 12.0),
            "H LITERAL, no explicit arm": (etr5, etr5, int2, False, False, None),
            "I LITERAL, arm but no re-arm after exit": (etr5, etr5, int2, True, False, None),
        }
        for k, (et, xt, tg, ra, rr, cap) in specs.items():
            t = simulate(d, et, xt, tg, ra, rr, cap)
            if len(t):
                t["symbol"] = f.stem
                combos.setdefault(k, []).append(t)

    rows = {}
    for k, v in combos.items():
        p = pd.concat(v)
        rows[k] = summarize(p)
        p.to_csv(Path(__file__).parent / f"dec_{k[0]}.csv", index=False)
    out = pd.DataFrame(rows).T
    pd.set_option("display.width", 300, "display.max_columns", 60)
    print(out.to_string())
    out.to_csv(Path(__file__).parent / "decompose_summary.csv")
