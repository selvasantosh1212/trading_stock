"""Part 4: non-bull-market tests.
  (i) 2018-01-01 .. 2020-12-31 sub-window on the original 5 (includes the
      2018 NBFC/IL&FS credit scare, a long 2019 sideways/grind, and the
      2020 COVID crash+recovery -- not a clean one-way bull run).
  (ii) Full 10y test on YESBANK.NS (collapsed >95%) and ONGC.NS (multi-year
       flat/declining), stocks that did NOT go up 5-10x, to check for
       survivorship bias in the original 5-stock sample.
  (iii) Full-universe (33 liquid NSE names across sectors) summary, to see
        whether the original 5's results are representative or cherry-picked.
"""
import glob
import os

import pandas as pd

from strategy import run_backtest, trades_to_df, summarize

pd.set_option("display.width", 160)

ORIGINAL_5 = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
NON_BULL = ["YESBANK.NS", "ONGC.NS"]


def load(sym):
    return pd.read_parquet(f"investigations/data/{sym.replace('.', '_')}_1d.parquet")


def buy_and_hold_pct(df):
    return (df["close"].iloc[-1] / df["close"].iloc[0] - 1.0) * 100.0


print("=" * 80)
print("(i) 2018-01-01 .. 2020-12-31 sub-window, original 5 stocks")
print("=" * 80)
all_trades = []
for sym in ORIGINAL_5:
    daily = load(sym)
    sub = daily.loc["2016-07-01":"2020-12-31"]  # extra lookback so pivots/htf trend exist by 2018
    trades = run_backtest(sub, entry_fill="next_open", stop_mode="none")
    tdf = trades_to_df(trades)
    # keep only trades whose ENTRY falls inside the actual 2018-2020 window
    tdf = tdf[(tdf["entry_time"] >= "2018-01-01") & (tdf["entry_time"] <= "2020-12-31")]
    tdf["symbol"] = sym
    all_trades.append(tdf)
    window_bh = daily.loc["2018-01-01":"2020-12-31"]
    bh = buy_and_hold_pct(window_bh)
    s = summarize(tdf)
    print(f"{sym:14s} n={s.get('n_trades',0):3d}  win_rate={s.get('win_rate',float('nan')):5.1f}%  "
          f"avg_ret={s.get('avg_ret_pct',float('nan')):6.2f}%  buy&hold(window)={bh:7.1f}%")
combined = pd.concat(all_trades, ignore_index=True)
s = summarize(combined)
print(f"\n-- combined 2018-2020 sub-window -- n={s.get('n_trades',0)}  win_rate={s.get('win_rate',float('nan')):.1f}%  "
      f"avg_ret={s.get('avg_ret_pct',float('nan')):.2f}%  worst={s.get('worst_trade_pct',float('nan')):.2f}%  "
      f"max_dd={s.get('max_drawdown_pct',float('nan')):.2f}%")


print("\n" + "=" * 80)
print("(ii) Full 10y test on non-bull-run stocks (survivorship-bias check)")
print("=" * 80)
for sym in NON_BULL:
    daily = load(sym)
    trades = run_backtest(daily, entry_fill="next_open", stop_mode="none")
    tdf = trades_to_df(trades)
    bh = buy_and_hold_pct(daily)
    s = summarize(tdf)
    print(f"\n{sym}: buy&hold over full window = {bh:.1f}%")
    print(f"  n={s.get('n_trades',0)}  win_rate={s.get('win_rate',float('nan')):.1f}%  "
          f"avg_ret={s.get('avg_ret_pct',float('nan')):.2f}%  worst={s.get('worst_trade_pct',float('nan')):.2f}%  "
          f"max_dd={s.get('max_drawdown_pct',float('nan')):.2f}%  total_compounded={s.get('total_compounded_pct',float('nan')):.1f}%")
    if len(tdf):
        print(tdf[["symbol"] if "symbol" in tdf.columns else []].shape if False else tdf.assign(symbol=sym)[["entry_time","exit_time","exit_reason","ret_pct"]].to_string(index=False))


print("\n" + "=" * 80)
print("(iii) Full universe (33 NSE stocks, all sectors) -- full 10y window")
print("=" * 80)
files = sorted(glob.glob("investigations/data/*_1d.parquet"))
universe_summaries = []
for fp in files:
    sym = os.path.basename(fp).replace("_1d.parquet", "").replace("_", ".", 1) if False else os.path.basename(fp).replace("_1d.parquet", "")
    daily = pd.read_parquet(fp)
    trades = run_backtest(daily, entry_fill="next_open", stop_mode="none")
    tdf = trades_to_df(trades)
    bh = buy_and_hold_pct(daily)
    s = summarize(tdf)
    universe_summaries.append({
        "symbol": sym,
        "n_trades": s.get("n_trades", 0),
        "win_rate": s.get("win_rate", float("nan")),
        "avg_ret_pct": s.get("avg_ret_pct", float("nan")),
        "total_compounded_pct": s.get("total_compounded_pct", float("nan")),
        "buy_hold_pct": bh,
        "strategy_beats_bh": (s.get("total_compounded_pct", float("-inf")) > bh) if s.get("n_trades", 0) else None,
    })

udf = pd.DataFrame(universe_summaries)
print(udf.to_string(index=False))
print(f"\nTotal stocks tested: {len(udf)}")
print(f"Stocks where strategy total-compounded return BEAT buy-and-hold: {udf['strategy_beats_bh'].sum()} / {udf['strategy_beats_bh'].notna().sum()}")
print(f"Median buy-and-hold over window: {udf['buy_hold_pct'].median():.1f}%   Median strategy total-compounded: {udf['total_compounded_pct'].median():.1f}%")
print(f"Overall avg win_rate across stocks with >=1 trade: {udf.loc[udf['n_trades']>0,'win_rate'].mean():.1f}%")
print(f"Overall avg avg_ret_pct across stocks with >=1 trade: {udf.loc[udf['n_trades']>0,'avg_ret_pct'].mean():.2f}%")
