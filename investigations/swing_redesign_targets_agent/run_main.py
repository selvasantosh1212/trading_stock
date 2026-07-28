import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from target_lab import load_all, run_symbol, summarize  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

VARIANTS = ("baseline", "t1t2", "t1run", "t1t2_be", "t1run_be", "t1run_trail")

data = load_all()
print(f"loaded {len(data)} symbols")

frames = []
for sym, df in data.items():
    t = run_symbol(sym, df, direction="long", t1_min_r=1.0, t2_min_r=2.0, variants=VARIANTS)
    if len(t):
        frames.append(t)
trades = pd.concat(frames, ignore_index=True)
trades.to_csv(Path(__file__).parent / "trades_long_main.csv", index=False)

print(f"\n=== SAMPLE: {len(trades)} long trades across {trades.symbol.nunique()} stocks "
      f"({trades.signal_date.min().date()} -> {trades.signal_date.max().date()}) ===")
print(f"still open at end of data: {int(trades.baseline_open.sum())}")
print(f"trades with NO T1 found (blue-sky, >=1R untaken level absent): "
      f"{int(trades.t1.isna().sum())} ({trades.t1.isna().mean()*100:.1f}%)")
print(f"median risk (stop distance) as % of entry: {trades.risk_pct.median():.2f}%")
print(f"nearest untaken level distance in R  -- median {trades.nearest_pool_r.median():.2f}, "
      f"pct below 1R: {(trades.nearest_pool_r < 1).mean()*100:.1f}%")
print("\nT1 distance R:", trades.t1_r.describe().round(2).to_dict())
print("T2 distance R:", trades.t2_r.describe().round(2).to_dict())
print("\nT1 level kind:\n", trades.t1_kind.value_counts(dropna=False))
print("\nT2 level kind:\n", trades.t2_kind.value_counts(dropna=False))

print("\n=== VARIANT COMPARISON (pooled, R-multiples) ===")
rowsum = []
for v in VARIANTS:
    s = summarize(trades, f"{v}_r")
    s["variant"] = v
    s["t1_hit_rate"] = round(trades[f"{v}_hit_t1"].mean() * 100, 1)
    s["t2_hit_rate"] = round(trades[f"{v}_hit_t2"].mean() * 100, 1)
    s["median_hold_days"] = int(trades[f"{v}_bars"].median())
    rowsum.append(s)
summ = pd.DataFrame(rowsum).set_index("variant")
print(summ)
summ.to_csv(Path(__file__).parent / "variant_summary_long.csv")

print("\n=== EXIT REASON MIX ===")
for v in VARIANTS:
    print(v, dict(trades[f"{v}_reason"].value_counts()))

print("\n=== T1/T2 hit conditional on a T1 actually existing ===")
has = trades[trades.t1.notna()]
print(f"n with T1 defined: {len(has)}")
for v in VARIANTS[1:]:
    print(f"  {v:14s} T1 hit {has[f'{v}_hit_t1'].mean()*100:5.1f}%   "
          f"T2 hit {has[f'{v}_hit_t2'].mean()*100:5.1f}%   "
          f"exp {has[f'{v}_r'].mean():+.3f}R vs baseline {has['baseline_r'].mean():+.3f}R")

print("\n=== PER-STOCK: t1run vs baseline expectancy ===")
per = trades.groupby("symbol").agg(n=("baseline_r", "size"),
                                   base=("baseline_r", "mean"),
                                   t1run=("t1run_r", "mean"),
                                   t1t2=("t1t2_r", "mean"),
                                   t1run_be=("t1run_be_r", "mean")).round(3)
per["t1run_beats_base"] = per.t1run > per.base
print(per)
print("\nstocks where t1run beats baseline:", int(per.t1run_beats_base.sum()), "/", len(per))
print("stocks where t1t2 beats baseline:", int((per.t1t2 > per.base).sum()), "/", len(per))
print("stocks where t1run_be beats baseline:", int((per.t1run_be > per.base).sum()), "/", len(per))
