"""Part 3: stop-loss overlay comparison -- none vs structure-stop (most
recent confirmed LTF swing low at entry) vs fixed 6% stop. Uses the
realistic next_open fill throughout.
"""
import pandas as pd

from strategy import run_backtest, trades_to_df, summarize

pd.set_option("display.width", 160)

ORIGINAL_5 = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]


def load(sym):
    return pd.read_parquet(f"investigations/data/{sym.replace('.', '_')}_1d.parquet")


if __name__ == "__main__":
    variants = [
        ("none", {}),
        ("structure", {}),
        ("pct", {"stop_pct": 0.06}),
        ("pct", {"stop_pct": 0.08}),
    ]

    for mode, kwargs in variants:
        label = mode if mode != "pct" else f"pct_{int(kwargs['stop_pct']*100)}pct"
        all_trades = []
        for sym in ORIGINAL_5:
            daily = load(sym)
            trades = run_backtest(daily, entry_fill="next_open", stop_mode=mode, **kwargs)
            tdf = trades_to_df(trades)
            tdf["symbol"] = sym
            all_trades.append(tdf)
        combined = pd.concat(all_trades, ignore_index=True)
        s = summarize(combined)
        stop_exits = (combined["exit_reason"] == "stop").sum() if len(combined) else 0
        print(f"\n--- stop_mode={label} ---")
        print(f"  n_trades={s.get('n_trades',0)}  win_rate={s.get('win_rate',float('nan')):.1f}%  "
              f"avg_ret={s.get('avg_ret_pct',float('nan')):.2f}%  worst_trade={s.get('worst_trade_pct',float('nan')):.2f}%  "
              f"max_dd={s.get('max_drawdown_pct',float('nan')):.2f}%  total_compounded={s.get('total_compounded_pct',float('nan')):.1f}%  "
              f"stop_exits={stop_exits}/{s.get('n_trades',0)}")
        if len(combined):
            worst5 = combined.nsmallest(5, "ret_pct")[["symbol", "entry_time", "exit_time", "exit_reason", "ret_pct"]]
            print("  worst 5 trades:")
            print(worst5.to_string(index=False))
