"""Part 1 & 3: Weekly-HTF / Daily-LTF backtest across the full 35-stock
universe, at the default pivot params (5,5 HTF / 5,5 LTF), with risk-adjusted
comparison against buy & hold (time-in-market, Sharpe-like ratio, max
drawdown).

Run: PYTHONPATH=. python investigations/run_full_universe.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from investigations.strategy import (
    DEFAULT_PARAMS,
    build_equity_curve,
    build_structures,
    buy_and_hold_curve,
    generate_trades,
    max_drawdown_pct,
    time_in_market_pct,
)
from investigations.universe import SECTOR_MAP, SYMBOLS

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_one(symbol: str, params=DEFAULT_PARAMS):
    fp = DATA_DIR / f"{symbol.replace('.', '_')}_1d.parquet"
    df = pd.read_parquet(fp)

    htf_struct, ltf_struct, htf_trend_on_ltf = build_structures(df, params)
    trades, open_trade = generate_trades(df, htf_trend_on_ltf, ltf_struct)

    eq = build_equity_curve(df, trades, open_trade)
    bh = buy_and_hold_curve(df)

    closed = trades  # closed trades only, for win-rate / avg-return stats
    n_closed = len(closed)
    if n_closed > 0:
        wins = closed["return_pct"] > 0
        win_rate = float(wins.mean() * 100)
        avg_ret = float(closed["return_pct"].mean())
        median_ret = float(closed["return_pct"].median())
        std_ret = float(closed["return_pct"].std(ddof=0)) if n_closed > 1 else float("nan")
        sharpe_like = avg_ret / std_ret if std_ret and not np.isnan(std_ret) and std_ret != 0 else float("nan")
    else:
        win_rate = avg_ret = median_ret = std_ret = sharpe_like = float("nan")

    years = (df.index[-1] - df.index[0]).days / 365.25

    return {
        "symbol": symbol,
        "sector": SECTOR_MAP.get(symbol, "?"),
        "years": round(years, 1),
        "n_trades_closed": n_closed,
        "n_trades_incl_open": n_closed + (1 if open_trade is not None else 0),
        "win_rate_pct": win_rate,
        "avg_return_pct": avg_ret,
        "median_return_pct": median_ret,
        "std_return_pct": std_ret,
        "sharpe_like": sharpe_like,
        "strategy_total_return_pct": float((eq.iloc[-1] - 1) * 100),
        "strategy_max_dd_pct": max_drawdown_pct(eq),
        "time_in_market_pct": time_in_market_pct(df, trades, open_trade),
        "buyhold_total_return_pct": float((bh.iloc[-1] - 1) * 100),
        "buyhold_max_dd_pct": max_drawdown_pct(bh),
        "has_open_trade": open_trade is not None,
    }


def main():
    rows = []
    for sym in SYMBOLS:
        fp = DATA_DIR / f"{sym.replace('.', '_')}_1d.parquet"
        if not fp.exists():
            print(f"skip {sym}: no cached data")
            continue
        rows.append(run_one(sym))

    df = pd.DataFrame(rows).sort_values("sector")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 100)
    print(df.to_string(index=False))

    out_fp = RESULTS_DIR / "full_universe_default_params.csv"
    df.to_csv(out_fp, index=False)
    print(f"\nSaved -> {out_fp}")

    # ---- aggregate stats (trade-weighted, i.e. pool all closed trades) ----
    all_trades = []
    for sym in SYMBOLS:
        fp = DATA_DIR / f"{sym.replace('.', '_')}_1d.parquet"
        if not fp.exists():
            continue
        d = pd.read_parquet(fp)
        htf_struct, ltf_struct, htf_trend_on_ltf = build_structures(d, DEFAULT_PARAMS)
        trades, _ = generate_trades(d, htf_trend_on_ltf, ltf_struct)
        if len(trades) > 0:
            trades = trades.copy()
            trades["symbol"] = sym
            all_trades.append(trades)
    pooled = pd.concat(all_trades, ignore_index=True)
    pooled.to_csv(RESULTS_DIR / "full_universe_pooled_trades.csv", index=False)

    print("\n=== POOLED TRADE-LEVEL STATS (all closed trades, all stocks) ===")
    print(f"n_trades = {len(pooled)}")
    print(f"win_rate = {(pooled['return_pct'] > 0).mean() * 100:.1f}%")
    print(f"avg_return = {pooled['return_pct'].mean():.2f}%")
    print(f"median_return = {pooled['return_pct'].median():.2f}%")
    print(f"std_return = {pooled['return_pct'].std():.2f}%")
    print(f"sharpe_like (mean/std) = {pooled['return_pct'].mean() / pooled['return_pct'].std():.3f}")
    print(f"best/worst = {pooled['return_pct'].max():.1f}% / {pooled['return_pct'].min():.1f}%")

    print("\n=== PER-STOCK SUMMARY STATS (unweighted across stocks) ===")
    numeric_cols = [
        "win_rate_pct", "avg_return_pct", "median_return_pct", "strategy_total_return_pct",
        "strategy_max_dd_pct", "time_in_market_pct", "buyhold_total_return_pct", "buyhold_max_dd_pct",
    ]
    print(df[numeric_cols].mean(numeric_only=True))

    print("\n=== BY SECTOR (mean strategy total return vs buy&hold total return) ===")
    sector_summary = df.groupby("sector")[["strategy_total_return_pct", "buyhold_total_return_pct", "n_trades_closed"]].mean()
    print(sector_summary)

    beats_bh = (df["strategy_total_return_pct"] > df["buyhold_total_return_pct"]).sum()
    print(f"\nStocks where strategy beats buy&hold on total return: {beats_bh}/{len(df)}")
    lower_dd = (df["strategy_max_dd_pct"] > df["buyhold_max_dd_pct"]).sum()  # less negative = shallower dd
    print(f"Stocks where strategy has SHALLOWER max drawdown than buy&hold: {lower_dd}/{len(df)}")

    # Return-per-unit-of-drawdown ("Calmar-like") ratio, to directly answer
    # "does the lower drawdown justify the lower total return?"
    df["strategy_return_over_dd"] = df["strategy_total_return_pct"] / df["strategy_max_dd_pct"].abs()
    df["buyhold_return_over_dd"] = df["buyhold_total_return_pct"] / df["buyhold_max_dd_pct"].abs()
    better_calmar = (df["strategy_return_over_dd"] > df["buyhold_return_over_dd"]).sum()
    print(f"\nStocks where strategy has a BETTER return/max-drawdown ratio than buy&hold: {better_calmar}/{len(df)}")
    print(f"Mean strategy return/dd ratio: {df['strategy_return_over_dd'].mean():.2f}")
    print(f"Mean buy&hold return/dd ratio: {df['buyhold_return_over_dd'].mean():.2f}")
    df.to_csv(out_fp, index=False)  # re-save with the two new ratio columns


if __name__ == "__main__":
    main()
