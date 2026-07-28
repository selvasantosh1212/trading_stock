"""Sensitivity: how far out T1 must be, and how much to take off at T1."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from target_lab import load_all, run_symbol, summarize  # noqa: E402

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 60)

data = load_all()


def fixed_frac(r, f):
    """Terminal equity multiple risking fraction f of capital per trade."""
    return float(np.prod(1.0 + f * r.to_numpy()))


def dd_of(r):
    eq = r.cumsum()
    return float((eq - eq.cummax()).min())


print("### PART A: T1 minimum-R sweep (runner uncapped, stop unchanged) ###")
rows = []
for t1r in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
    frames = []
    for sym, df in data.items():
        t = run_symbol(sym, df, "long", t1_min_r=t1r, t2_min_r=t1r * 2,
                       variants=("baseline", "t1run", "t1t2"))
        if len(t):
            frames.append(t)
    tr = pd.concat(frames, ignore_index=True)
    base = summarize(tr, "baseline_r")
    for v in ("t1run", "t1t2"):
        s = summarize(tr, f"{v}_r")
        rows.append({
            "t1_min_r": t1r, "variant": v, "n": s["n"],
            "pct_no_t1": round(tr.t1.isna().mean() * 100, 1),
            "t1_hit%": round(tr[f"{v}_hit_t1"].mean() * 100, 1),
            "win%": s["win_rate"], "exp_R": s["expectancy_r"], "med_R": s["median_r"],
            "PF": s["profit_factor"], "maxDD_R": s["max_dd_r"],
            "eq@2%": round(fixed_frac(tr[f"{v}_r"], 0.02), 2),
            "eq@5%": round(fixed_frac(tr[f"{v}_r"], 0.05), 2),
        })
    if t1r == 0.5:
        rows.append({"t1_min_r": "-", "variant": "BASELINE", "n": base["n"], "pct_no_t1": 0.0,
                     "t1_hit%": 0.0, "win%": base["win_rate"], "exp_R": base["expectancy_r"],
                     "med_R": base["median_r"], "PF": base["profit_factor"],
                     "maxDD_R": base["max_drawdown_r"] if "max_drawdown_r" in base else base["max_dd_r"],
                     "eq@2%": round(fixed_frac(tr["baseline_r"], 0.02), 2),
                     "eq@5%": round(fixed_frac(tr["baseline_r"], 0.05), 2)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n### PART B: partial-exit SIZE at T1 (T1>=1.5R, runner uncapped, stop unchanged) ###")
frames = []
for sym, df in data.items():
    t = run_symbol(sym, df, "long", t1_min_r=1.5, t2_min_r=3.0, variants=("baseline", "t1run"))
    if len(t):
        frames.append(t)
tr = pd.concat(frames, ignore_index=True)
# reconstruct arbitrary splits from the 50/50 legs:
#   total_r(f) = f*leg1 + (1-f)*leg2 ; and total_r(0.5) = 0.5*leg1+0.5*leg2
# we need the legs, so recompute quickly using the stored per-variant pieces is
# not possible -> derive legs by re-running with two extreme splits instead.
# Simpler: leg1 = t1_r when hit; leg2 = 2*total - leg1.
hit = tr["t1run_hit_t1"]
leg1 = np.where(hit, tr["t1_r"], tr["t1run_r"])
leg2 = np.where(hit, 2 * tr["t1run_r"] - leg1, tr["t1run_r"])
out = []
for f in [0.0, 0.25, 0.33, 0.5, 0.67, 0.75, 1.0]:
    r = pd.Series(np.where(hit, f * leg1 + (1 - f) * leg2, tr["t1run_r"]))
    s = summarize(pd.DataFrame({"r": r}), "r")
    out.append({"frac_off_at_T1": f, "win%": s["win_rate"], "exp_R": s["expectancy_r"],
                "med_R": s["median_r"], "PF": s["profit_factor"], "maxDD_R": s["max_dd_r"],
                "eq@2%": round(fixed_frac(r, 0.02), 2), "eq@5%": round(fixed_frac(r, 0.05), 2)})
print(pd.DataFrame(out).to_string(index=False))

print("\n### PART C: tail structure of the baseline (why targets hurt) ###")
b = tr["baseline_r"].sort_values(ascending=False)
print("top 10 baseline trades (R):", [round(x, 1) for x in b.head(10)])
print(f"share of total baseline R from top 5 trades: {b.head(5).sum()/b.sum()*100:.1f}%")
print(f"share of total baseline R from top 10 trades: {b.head(10).sum()/b.sum()*100:.1f}%")
print(f"trades with baseline R > 5: {(b>5).sum()} / {len(b)} ({(b>5).mean()*100:.1f}%)")
print(f"of those, T1 was hit first in: {int(tr.loc[tr.baseline_r>5,'t1run_hit_t1'].sum())}")

print("\n### PART D: SHORT side (mirrored strategy, currently unimplemented) ###")
frames = []
for sym, df in data.items():
    t = run_symbol(sym, df, "short", t1_min_r=1.5, t2_min_r=3.0, variants=("baseline", "t1run", "t1t2"))
    if len(t):
        frames.append(t)
sh = pd.concat(frames, ignore_index=True)
sh.to_csv(Path(__file__).parent / "trades_short.csv", index=False)
for v in ("baseline", "t1run", "t1t2"):
    s = summarize(sh, f"{v}_r")
    print(f"  SHORT {v:9s}", s)
print(f"  short trades: {len(sh)} across {sh.symbol.nunique()} stocks; "
      f"median hold {int(sh.baseline_bars.median())}d; median risk {sh.risk_pct.median():.2f}%")
