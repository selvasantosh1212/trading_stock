"""Swing profit-target design lab (RESEARCH ONLY - writes nothing to the repo).

Attaches candidate liquidity-based profit targets to the EXISTING, already
validated stock_hh_ll_tool/entry_exit.py entry signal (Weekly HTF bullish
trend + Daily LTF bullish CHoCH, filled at the NEXT day's open, structure
stop = daily swing low at signal - 0.1 ATR) and measures them in R-multiples
against the current no-target baseline (stop / weekly-trend-reversal only).

Causality rules enforced everywhere:
  * every target level is derived ONLY from bars with index <= signal bar i
  * a fractal pivot is only usable from its `confirmed_at` bar onward
  * fill is at bar i+1's open; stop/target hit-scanning starts at bar i+2
    (matches investigations/strategy.py's `if i > entry_bar` guard), so no
    bar ever both defines and resolves a level
"""
from __future__ import annotations

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
HTF_L = HTF_R = 5
LTF_L = LTF_R = 3
STOP_ATR_FRAC = 0.1


# --------------------------------------------------------------------------
# level pool
# --------------------------------------------------------------------------
def build_level_pool(daily: pd.DataFrame, weekly: pd.DataFrame, direction: str):
    """Returns a list of (usable_from_bar_idx, level_price, origin_bar_idx, kind)
    for every liquidity level this design considers. `usable_from_bar_idx` is
    the first DAILY bar index at which the level is causally knowable.

    kinds: 'daily_swing' (LTF 3/3 fractal pivot), 'weekly_swing' (HTF 5/5
    fractal pivot on weekly bars), 'prev_month' (previous completed calendar
    month's extreme).
    """
    idx = daily.index
    n = len(daily)
    hi = daily["high"].to_numpy()
    lo = daily["low"].to_numpy()
    px = hi if direction == "long" else lo
    pivot_col = "pivot_high" if direction == "long" else "pivot_low"

    levels = []

    # --- daily fractal pivots -------------------------------------------
    dp = detect_fractal_pivots(daily, LTF_L, LTF_R)
    flags = dp[pivot_col].to_numpy()
    conf = dp["confirmed_at"].to_numpy()
    for p in range(n):
        if flags[p]:
            c = int(conf[p])
            if c < n:
                levels.append((c, float(px[p]), p, "daily_swing"))

    # --- weekly fractal pivots, mapped onto the daily index --------------
    wp = detect_fractal_pivots(weekly, HTF_L, HTF_R)
    wflags = wp[pivot_col].to_numpy()
    wconf = wp["confirmed_at"].to_numpy()
    wpx = (weekly["high"] if direction == "long" else weekly["low"]).to_numpy()
    wts = weekly.index
    for p in range(len(weekly)):
        if wflags[p]:
            c = int(wconf[p])
            if c >= len(weekly):
                continue
            # weekly bars are stamped at their CLOSE (resample label="right"),
            # so the confirming weekly bar's timestamp is the moment the level
            # becomes knowable. Map to the first daily bar at/after that.
            pos = idx.searchsorted(wts[c], side="left")
            if pos < n:
                # origin daily bar = last daily bar of the pivot's own week
                opos = int(idx.searchsorted(wts[p], side="left")) - 1
                levels.append((int(pos), float(wpx[p]), max(opos, 0), "weekly_swing"))

    # --- previous completed calendar month's extreme ---------------------
    naive = idx.tz_convert("UTC").tz_localize(None)
    per = pd.Series(naive.to_period("M"), index=idx)
    m_ext = daily.groupby(per)["high" if direction == "long" else "low"].agg(
        "max" if direction == "long" else "min"
    )
    per_list = list(m_ext.index)
    for k in range(1, len(per_list)):
        prev_p = per_list[k - 1]
        # first daily bar of month k is when the previous month's extreme is known
        first_bar = int(np.argmax((per == per_list[k]).to_numpy()))
        last_bar_prev = first_bar - 1
        levels.append((first_bar, float(m_ext.iloc[k - 1]), max(last_bar_prev, 0), "prev_month"))

    levels.sort(key=lambda t: t[0])
    return levels


def untaken_levels_at(levels, i, hi, lo, direction, entry):
    """All levels knowable at bar i, still UNTAKEN (price has not wicked
    through them since the level's origin bar), and beyond `entry`.

    'Taken' uses the bar HIGH/LOW, not the close: stops resting above a prior
    high are triggered by the trade price, so a wick through the level has
    already consumed that liquidity pool. This is deliberately stricter than
    bos_choch.py's close-only break rule (see report).
    """
    out = []
    for usable_from, price, origin, kind in levels:
        if usable_from > i:
            continue
        if origin + 1 > i:
            continue
        if direction == "long":
            if price <= entry:
                continue
            if hi[origin + 1 : i + 1].max(initial=-np.inf) >= price:
                continue
        else:
            if price >= entry:
                continue
            if lo[origin + 1 : i + 1].min(initial=np.inf) <= price:
                continue
        out.append((price, kind))
    if direction == "long":
        out.sort(key=lambda t: t[0])
    else:
        out.sort(key=lambda t: -t[0])
    return out


def pick_targets(pool, entry, risk, direction, t1_min_r, t2_min_r):
    sign = 1.0 if direction == "long" else -1.0
    t1 = t1_kind = t2 = t2_kind = None
    for price, kind in pool:
        r = sign * (price - entry) / risk
        if r >= t1_min_r:
            t1, t1_kind = price, kind
            break
    if t1 is not None:
        for price, kind in pool:
            r = sign * (price - entry) / risk
            if r >= t2_min_r and sign * (price - t1) > 0:
                t2, t2_kind = price, kind
                break
    return t1, t1_kind, t2, t2_kind


# --------------------------------------------------------------------------
# simulation
# --------------------------------------------------------------------------
def run_symbol(sym, daily, direction="long", t1_min_r=1.0, t2_min_r=2.0,
               variants=("baseline",), htf_freq="W"):
    weekly = resample_ohlc(daily, htf_freq)
    htf_struct = compute_structure(weekly, detect_fractal_pivots(weekly, HTF_L, HTF_R), close_only=True)
    ltf_struct = compute_structure(daily, detect_fractal_pivots(daily, LTF_L, LTF_R), close_only=True)
    htf_trend = align_htf_to_ltf(htf_struct["trend"], daily.index).to_numpy()

    n = len(daily)
    idx = daily.index
    op, hi, lo, cl = (daily[c].to_numpy() for c in ("open", "high", "low", "close"))
    ev = ltf_struct["event"].to_numpy()
    evd = ltf_struct["event_direction"].to_numpy()
    swing_low = ltf_struct["swing_low"].to_numpy()
    swing_high = ltf_struct["swing_high"].to_numpy()
    atr_s = atr(daily).to_numpy()

    levels = build_level_pool(daily, weekly, direction)

    want_trend = "bullish" if direction == "long" else "bearish"
    opp_trend = "bearish" if direction == "long" else "bullish"
    sign = 1.0 if direction == "long" else -1.0

    rows = []
    i = 0
    while i < n - 2:
        if not (htf_trend[i] == want_trend and ev[i] == "CHoCH" and evd[i] == want_trend):
            i += 1
            continue
        anchor = swing_low[i] if direction == "long" else swing_high[i]
        if pd.isna(anchor):
            i += 1
            continue
        entry_i = i + 1
        entry = float(op[entry_i])
        buf = STOP_ATR_FRAC * atr_s[i] if not np.isnan(atr_s[i]) else 0.0
        stop0 = float(anchor) - buf if direction == "long" else float(anchor) + buf
        risk = sign * (entry - stop0)
        if risk <= 0:
            i = entry_i + 1
            continue

        pool = untaken_levels_at(levels, i, hi, lo, direction, entry)
        t1, t1_kind, t2, t2_kind = pick_targets(pool, entry, risk, direction, t1_min_r, t2_min_r)

        res = {v: simulate(v, daily, entry_i, entry, stop0, risk, t1, t2, htf_trend,
                           opp_trend, ltf_struct, direction, sign, op, hi, lo, cl, n)
               for v in variants}
        # exit bar of the baseline drives when the next signal may be taken
        exit_bar = res["baseline"]["exit_bar"]

        row = {
            "symbol": sym, "signal_date": idx[i], "entry_date": idx[entry_i],
            "entry": entry, "stop": stop0, "risk_pct": risk / entry * 100.0,
            "t1": t1, "t1_kind": t1_kind, "t2": t2, "t2_kind": t2_kind,
            "t1_r": (sign * (t1 - entry) / risk) if t1 is not None else np.nan,
            "t2_r": (sign * (t2 - entry) / risk) if t2 is not None else np.nan,
            "n_pool": len(pool),
            "nearest_pool_r": (sign * (pool[0][0] - entry) / risk) if pool else np.nan,
        }
        for v, r in res.items():
            row[f"{v}_r"] = r["total_r"]
            row[f"{v}_reason"] = r["reason"]
            row[f"{v}_hit_t1"] = r["hit_t1"]
            row[f"{v}_hit_t2"] = r["hit_t2"]
            row[f"{v}_bars"] = r["exit_bar"] - entry_i
            row[f"{v}_open"] = r["unresolved"]
        rows.append(row)
        i = max(exit_bar, entry_i) + 1

    return pd.DataFrame(rows)


def simulate(variant, daily, entry_i, entry, stop0, risk, t1, t2, htf_trend,
             opp_trend, ltf_struct, direction, sign, op, hi, lo, cl, n):
    """variant grammar:
      baseline          : no target at all (current production behaviour)
      t1t2              : 50% at T1, 50% at T2 (hard cap), stop unchanged
      t1run             : 50% at T1, runner has NO price cap (stop/trend only)
      t1t2_be / t1run_be: same, stop -> breakeven after T1
      t1run_trail       : runner's stop trails the newest confirmed daily swing low
    """
    use_t1 = variant != "baseline" and t1 is not None
    cap_t2 = variant.startswith("t1t2") and t2 is not None
    be = variant.endswith("_be")
    trail = variant.endswith("_trail")

    swing_anchor = (ltf_struct["swing_low"] if direction == "long" else ltf_struct["swing_high"]).to_numpy()

    stop = stop0
    leg1_r = None
    leg2_r = None
    hit_t1 = hit_t2 = False
    reason = None
    exit_bar = n - 1
    unresolved = True

    j = entry_i + 1  # first bar the stop/target may act on
    while j < n:
        if direction == "long":
            stop_hit = lo[j] <= stop
            t1_hit = (t1 is not None) and hi[j] >= t1
            t2_hit = (t2 is not None) and hi[j] >= t2
        else:
            stop_hit = hi[j] >= stop
            t1_hit = (t1 is not None) and lo[j] <= t1
            t2_hit = (t2 is not None) and lo[j] <= t2

        # conservative same-bar precedence: stop resolves before any target
        if stop_hit:
            fill = op[j] if sign * (op[j] - stop) < 0 else stop
            r = sign * (fill - entry) / risk
            if not hit_t1 or not use_t1:
                leg1_r = leg2_r = r
                reason = "stop"
            else:
                leg2_r = r
                reason = "stop_after_t1"
            exit_bar, unresolved = j, False
            break

        if use_t1 and not hit_t1 and t1_hit:
            hit_t1 = True
            leg1_r = sign * (t1 - entry) / risk
            if be:
                stop = entry
            if cap_t2 and t2_hit:
                # same bar tagged both -> conservatively take T1 only on this
                # bar, let T2 resolve on a later bar
                pass
            j += 1
            continue

        if use_t1 and hit_t1 and cap_t2 and t2_hit:
            hit_t2 = True
            leg2_r = sign * (t2 - entry) / risk
            reason = "t2"
            exit_bar, unresolved = j, False
            break

        if trail and hit_t1:
            a = swing_anchor[j]
            if not pd.isna(a):
                cand = float(a)
                if sign * (cand - stop) > 0:
                    stop = cand

        if htf_trend[j] == opp_trend:
            if j + 1 >= n:
                break
            fill = op[j + 1]
            r = sign * (fill - entry) / risk
            if hit_t1 and use_t1:
                leg2_r = r
            else:
                leg1_r = leg2_r = r
            reason = "trend_exit"
            exit_bar, unresolved = j + 1, False
            break
        j += 1

    if unresolved:
        fill = cl[n - 1]
        r = sign * (fill - entry) / risk
        if hit_t1 and use_t1:
            leg2_r = r
        else:
            leg1_r = leg2_r = r
        reason = "open_at_end"
        exit_bar = n - 1

    if leg2_r is None:
        leg2_r = leg1_r
    total_r = 0.5 * leg1_r + 0.5 * leg2_r if (use_t1 and hit_t1) else leg1_r
    return {"total_r": total_r, "reason": reason, "hit_t1": hit_t1, "hit_t2": hit_t2,
            "exit_bar": exit_bar, "unresolved": unresolved}


def summarize(trades, col):
    r = trades[col].dropna()
    if len(r) == 0:
        return {}
    wins = r > 0
    gw = r[wins].sum()
    gl = -r[~wins].sum()
    eq = r.cumsum()
    dd = (eq - eq.cummax()).min()
    return {
        "n": len(r),
        "win_rate": round(float(wins.mean()) * 100, 1),
        "expectancy_r": round(float(r.mean()), 3),
        "median_r": round(float(r.median()), 3),
        "avg_win_r": round(float(r[wins].mean()), 3) if wins.any() else 0.0,
        "avg_loss_r": round(float(r[~wins].mean()), 3) if (~wins).any() else 0.0,
        "profit_factor": round(float(gw / gl), 2) if gl > 0 else float("inf"),
        "max_dd_r": round(float(dd), 2),
        "best_r": round(float(r.max()), 2),
    }


def load_all():
    out = {}
    for f in sorted(DATA.glob("*_1d.parquet")):
        sym = f.name.replace("_NS_1d.parquet", "")
        df = pd.read_parquet(f)
        out[sym] = df
    return out
