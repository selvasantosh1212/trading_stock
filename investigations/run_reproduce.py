"""Part 1: independently reproduce the original 5-stock backtest, and
directly compare the 'signal_close' (naive/optimistic fill) vs 'next_open'
(realistic fill) conventions to quantify part 2(b)'s execution-timing gap.
"""
import pandas as pd

from strategy import run_backtest, trades_to_df, summarize

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

ORIGINAL_5 = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]


def load(sym):
    fp = f"investigations/data/{sym.replace('.', '_')}_1d.parquet"
    df = pd.read_parquet(fp)
    return df


def buy_and_hold_pct(df):
    return (df["close"].iloc[-1] / df["close"].iloc[0] - 1.0) * 100.0


if __name__ == "__main__":
    for fill in ["signal_close", "next_open"]:
        print(f"\n{'=' * 80}\nENTRY/EXIT FILL CONVENTION: {fill}\n{'=' * 80}")
        all_trades = []
        for sym in ORIGINAL_5:
            daily = load(sym)
            trades = run_backtest(daily, entry_fill=fill, stop_mode="none")
            tdf = trades_to_df(trades)
            tdf["symbol"] = sym
            all_trades.append(tdf)
            bh = buy_and_hold_pct(daily)
            s = summarize(tdf)
            print(f"{sym:14s} n={s.get('n_trades',0):3d}  win_rate={s.get('win_rate',float('nan')):5.1f}%  "
                  f"avg_ret={s.get('avg_ret_pct',float('nan')):6.2f}%  buy&hold={bh:7.1f}%")

        combined = pd.concat(all_trades, ignore_index=True)
        overall = summarize(combined)
        print(f"\n-- COMBINED across {len(ORIGINAL_5)} stocks ({fill}) --")
        for k, v in overall.items():
            print(f"  {k}: {v}")
