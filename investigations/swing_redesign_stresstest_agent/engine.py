"""Independent stress-test reimplementation of a swing strategy with staged
liquidity profit targets. Written from the source primitives, NOT copied from
stock_hh_ll_tool/entry_exit.py or investigations/strategy.py.

Design (see report): Weekly HTF trend + Daily LTF CHoCH entry (the validated
pairing), structure stop, and NEW staged liquidity targets T1/T2 drawn from a
causally-built pool of untaken buy-side liquidity above price:
  - most recent confirmed daily swing high (the retracement's origin)
  - previous completed week's high
  - previous completed month's high
  - older confirmed daily swing highs never yet closed through
T1 = nearest such level above entry, T2 = next nearest. Sell HALF at T1, move
stop to breakeven, run the rest to T2 (or stop/weekly-trend exit) -- the Pine
script's stated partial-at-T1-then-run model, rescaled from
session/prev-day (intraday) to week/month (swing).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from smc_validator.data_ingestion.resample import align_htf_to_ltf, resample_ohlc
from smc_validator.liquidity.daily_weekly import previous_period_levels
from smc_validator.patterns.order_blocks import atr
from smc_validator.structure.bos_choch import compute_structure
from smc_validator.structure.swings import detect_fractal_pivots


# --------------------------------------------------------------------------
# Causal liquidity pool
# --------------------------------------------------------------------------
def build_untaken_swing_highs(daily: pd.DataFrame, pivots: pd.DataFrame) -> list[list[float]]:
    """untaken[i] = list of confirmed pivot-high PRICES that, as of bar i's
    close, have been confirmed (confirmed_at <= i) and have never had a close
    above them since confirmation. Strictly causal: a pivot only enters the
    pool on its confirmed_at bar, never on the bar it formed.
    """
    n = len(daily)
    high = daily["high"].to_numpy()
    close = daily["close"].to_numpy()
    ph = pivots["pivot_high"].to_numpy()
    conf = pivots["confirmed_at"]

    confirms: dict[int, list[float]] = {}
    for j in range(n):
        if ph[j]:
            confirms.setdefault(int(conf.iloc[j]), []).append(float(high[j]))

    pool: list[float] = []
    out: list[list[float]] = []
    for i in range(n):
        for price in confirms.get(i, []):
            pool.append(price)
        # a level a close has gone through is swept liquidity, not resting liquidity
        pool = [p for p in pool if p > close[i]]
        out.append(list(pool))
    return out


def liquidity_targets(candidates: list[float], ref_price: float, n_targets: int = 2) -> list[float]:
    """Nearest untaken levels strictly above ref_price, ascending."""
    above = sorted({round(c, 6) for c in candidates if c > ref_price})
    return above[:n_targets]


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------
def run(
    daily: pd.DataFrame,
    *,
    htf_left: int = 5,
    htf_right: int = 5,
    ltf_left: int = 3,
    ltf_right: int = 3,
    stop_atr_frac: float = 0.1,
    use_targets: bool = True,
    t1_fraction: float = 0.5,
    breakeven_after_t1: bool = True,
    same_bar_tie: str = "stop_first",
    trend_exit_fill: str = "next_open",
    mode: str = "weekly_daily",
    min_r_to_t1: float = 0.0,
    entry_bar_active: bool = True,
) -> dict:
    """Long-only. Returns {'trades': DataFrame, 'equity': Series}."""
    n = len(daily)
    idx = daily.index
    o = daily["open"].to_numpy()
    h = daily["high"].to_numpy()
    lo = daily["low"].to_numpy()
    c = daily["close"].to_numpy()

    if mode == "weekly_daily":
        weekly = resample_ohlc(daily, "W")
        htf_struct = compute_structure(weekly, detect_fractal_pivots(weekly, htf_left, htf_right), close_only=True)
        htf_trend = align_htf_to_ltf(htf_struct["trend"], idx).to_numpy()
    elif mode == "daily_internal_external":
        ext = compute_structure(daily, detect_fractal_pivots(daily, htf_left, htf_right), close_only=True)
        htf_trend = ext["trend"].to_numpy()
    else:
        raise ValueError(mode)

    ltf_pivots = detect_fractal_pivots(daily, ltf_left, ltf_right)
    ltf = compute_structure(daily, ltf_pivots, close_only=True)
    ltf_event = ltf["event"].to_numpy()
    ltf_dir = ltf["event_direction"].to_numpy()
    ltf_swing_low = ltf["swing_low"].to_numpy(dtype=float)
    ltf_swing_high = ltf["swing_high"].to_numpy(dtype=float)

    atr_d = atr(daily).to_numpy(dtype=float)
    wk = previous_period_levels(daily, "W")["prev_period_high"].to_numpy(dtype=float)
    mo = previous_period_levels(daily, "M")["prev_period_high"].to_numpy(dtype=float)
    untaken = build_untaken_swing_highs(daily, ltf_pivots)

    entry_ok = (htf_trend == "bullish") & (ltf_event == "CHoCH") & (ltf_dir == "bullish")
    exit_trend = htf_trend == "bearish"

    cash, shares = 1.0, 0.0
    equity = np.empty(n)
    trades = []

    in_pos = False
    st = {}

    def close_trade(reason, exit_i, exit_price):
        nonlocal cash, shares, in_pos, st
        cash += shares * exit_price
        shares = 0.0
        risk = st["entry_price"] - st["init_stop"]
        r_t1 = st["t1_r"]
        r_rest = (exit_price - st["entry_price"]) / risk * (1.0 - st["filled_frac"])
        trades.append(
            {
                "signal_time": idx[st["signal_i"]],
                "entry_time": idx[st["entry_i"]],
                "entry_price": st["entry_price"],
                "exit_time": idx[exit_i],
                "exit_price": exit_price,
                "exit_reason": reason,
                "init_stop": st["init_stop"],
                "t1": st["t1"],
                "t2": st["t2"],
                "t1_hit": st["filled_frac"] > 0,
                "t1_dist_r": (st["t1"] - st["entry_price"]) / risk if st["t1"] else np.nan,
                "t2_dist_r": (st["t2"] - st["entry_price"]) / risk if st["t2"] else np.nan,
                "total_r": r_t1 + r_rest,
                "net_return_pct": (cash / st["cash_at_entry"] - 1.0) * 100.0,
                "holding_days": exit_i - st["entry_i"],
            }
        )
        in_pos = False
        st = {}

    for i in range(n):
        closed_this_bar = False
        if in_pos and (i >= st["entry_i"] if entry_bar_active else i > st["entry_i"]):
            j = i
            # --- 0. a trend-exit signalled at yesterday's close fills at today's open
            if st.get("pending_trend_exit") == j:
                close_trade("trend", j, o[j])
                closed_this_bar = True
            if in_pos:
                stop = st["stop"]
                # --- 1. gap through stop at the open -> fill at the OPEN, not the stop
                if o[j] <= stop:
                    close_trade("stop_gap", j, o[j])
                    closed_this_bar = True
                elif lo[j] <= stop and same_bar_tie == "stop_first":
                    close_trade("stop", j, stop)
                    closed_this_bar = True
                else:
                    hit_t1 = (
                        use_targets
                        and st["filled_frac"] == 0.0
                        and st["t1"] is not None
                        and h[j] >= st["t1"]
                    )
                    if hit_t1:
                        fill = max(st["t1"], o[j])  # gap-up open past T1 fills at the open
                        part = shares * t1_fraction
                        cash += part * fill
                        shares -= part
                        risk = st["entry_price"] - st["init_stop"]
                        st["t1_r"] = (fill - st["entry_price"]) / risk * t1_fraction
                        st["filled_frac"] = t1_fraction
                        st["t1_fill"] = fill
                        if breakeven_after_t1:
                            st["stop"] = max(st["stop"], st["entry_price"])
                    if in_pos and use_targets and st["filled_frac"] > 0 and st["t2"] is not None and h[j] >= st["t2"]:
                        close_trade("t2", j, max(st["t2"], o[j]))
                        closed_this_bar = True
                    if in_pos and same_bar_tie == "target_first" and not hit_t1 and lo[j] <= st["stop"]:
                        close_trade("stop", j, st["stop"])
                        closed_this_bar = True
                    if in_pos and exit_trend[j]:
                        if trend_exit_fill == "next_open":
                            if j + 1 < n:
                                st["pending_trend_exit"] = j + 1
                        else:
                            close_trade("trend", j, c[j])
                            closed_this_bar = True

        if not in_pos and not closed_this_bar and i > 0 and entry_ok[i - 1]:
            si = i - 1  # signal bar; fill on THIS bar's open
            sl = ltf_swing_low[si]
            if not np.isnan(sl):
                a = atr_d[si]
                stop = float(sl) - (stop_atr_frac * a if not np.isnan(a) else 0.0)
                entry_price = o[i]
                risk = entry_price - stop
                if risk > 0:
                    cands = list(untaken[si])
                    if not np.isnan(ltf_swing_high[si]):
                        cands.append(float(ltf_swing_high[si]))
                    if not np.isnan(wk[si]):
                        cands.append(float(wk[si]))
                    if not np.isnan(mo[si]):
                        cands.append(float(mo[si]))
                    tg = liquidity_targets(cands, entry_price, 2)
                    t1 = tg[0] if len(tg) > 0 else None
                    t2 = tg[1] if len(tg) > 1 else None
                    if t1 is not None and min_r_to_t1 > 0 and (t1 - entry_price) / risk < min_r_to_t1:
                        tg = [x for x in liquidity_targets(cands, entry_price, 8)
                              if (x - entry_price) / risk >= min_r_to_t1]
                        t1 = tg[0] if len(tg) > 0 else None
                        t2 = tg[1] if len(tg) > 1 else None
                    shares = cash / entry_price
                    st = {
                        "signal_i": si, "entry_i": i, "entry_price": entry_price,
                        "init_stop": stop, "stop": stop, "t1": t1, "t2": t2,
                        "filled_frac": 0.0, "t1_r": 0.0, "cash_at_entry": cash,
                    }
                    cash = 0.0
                    in_pos = True

        equity[i] = cash + shares * c[i]

    tdf = pd.DataFrame(trades)
    return {"trades": tdf, "equity": pd.Series(equity, index=idx)}


def max_dd(eq: pd.Series) -> float:
    return float((eq / eq.cummax() - 1.0).min() * 100.0)
