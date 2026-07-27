"""Part 2: Parameter sensitivity test.

Runs the full 35-stock Weekly-HTF/Daily-LTF backtest at 3 symmetric pivot
lookback settings (small/default/large, applied to BOTH the HTF weekly
pivots and the LTF daily pivots) plus 2 asymmetric combos to check
cross-sensitivity, and reports aggregate performance for each so we can see
whether the strategy's edge (or lack thereof) is stable across reasonable
choices of "how significant a swing point must be" - or whether it only
"works" for one specific tuned setting (a red flag for curve-fitting).

Run: PYTHONPATH=. python investigations/run_sensitivity.py
"""
from pathlib import Path

import pandas as pd

from investigations.strategy import PivotParams, build_structures, generate_trades
from investigations.universe import SYMBOLS

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PARAM_GRID = [
    PivotParams(htf_left=3, htf_right=3, ltf_left=3, ltf_right=3),   # smaller
    PivotParams(htf_left=5, htf_right=5, ltf_left=5, ltf_right=5),   # default (matches config's external default)
    PivotParams(htf_left=8, htf_right=8, ltf_left=8, ltf_right=8),   # larger
    # cross-sensitivity checks: HTF and LTF lookback varied independently
    PivotParams(htf_left=8, htf_right=8, ltf_left=3, ltf_right=3),   # strict HTF, loose LTF
    PivotParams(htf_left=3, htf_right=3, ltf_left=8, ltf_right=8),   # loose HTF, strict LTF
]


def run_for_params(params: PivotParams) -> dict:
    all_trades = []
    per_stock_rows = []
    for sym in SYMBOLS:
        fp = DATA_DIR / f"{sym.replace('.', '_')}_1d.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        htf_struct, ltf_struct, htf_trend_on_ltf = build_structures(df, params)
        trades, _ = generate_trades(df, htf_trend_on_ltf, ltf_struct)
        if len(trades) > 0:
            trades = trades.copy()
            trades["symbol"] = sym
            all_trades.append(trades)
            per_stock_rows.append(
                {
                    "symbol": sym,
                    "n_trades": len(trades),
                    "win_rate": float((trades["return_pct"] > 0).mean() * 100),
                    "avg_return": float(trades["return_pct"].mean()),
                }
            )
        else:
            per_stock_rows.append({"symbol": sym, "n_trades": 0, "win_rate": float("nan"), "avg_return": float("nan")})

    pooled = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(columns=["return_pct"])
    per_stock = pd.DataFrame(per_stock_rows)

    n = len(pooled)
    if n > 0:
        win_rate = float((pooled["return_pct"] > 0).mean() * 100)
        avg_ret = float(pooled["return_pct"].mean())
        median_ret = float(pooled["return_pct"].median())
        std_ret = float(pooled["return_pct"].std())
        n_stocks_with_trades = int((per_stock["n_trades"] > 0).sum())
        n_stocks_profitable = int((per_stock["avg_return"] > 0).sum())
    else:
        win_rate = avg_ret = median_ret = std_ret = float("nan")
        n_stocks_with_trades = n_stocks_profitable = 0

    return {
        "params": params.label,
        "n_trades_pooled": n,
        "n_stocks_with_trades": n_stocks_with_trades,
        "n_stocks_profitable_avg": n_stocks_profitable,
        "win_rate_pct": win_rate,
        "avg_return_pct": avg_ret,
        "median_return_pct": median_ret,
        "std_return_pct": std_ret,
    }


def main():
    rows = [run_for_params(p) for p in PARAM_GRID]
    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    out_fp = RESULTS_DIR / "sensitivity_summary.csv"
    summary.to_csv(out_fp, index=False)
    print(f"\nSaved -> {out_fp}")

    print("\n--- Stability check across the 3 symmetric settings (small/default/large) ---")
    sym_only = summary.iloc[:3]
    print(f"win_rate range: {sym_only['win_rate_pct'].min():.1f}% - {sym_only['win_rate_pct'].max():.1f}%")
    print(f"avg_return range: {sym_only['avg_return_pct'].min():.2f}% - {sym_only['avg_return_pct'].max():.2f}%")
    print(f"median_return range: {sym_only['median_return_pct'].min():.2f}% - {sym_only['median_return_pct'].max():.2f}%")
    print(f"n_trades range: {sym_only['n_trades_pooled'].min()} - {sym_only['n_trades_pooled'].max()}")


if __name__ == "__main__":
    main()
