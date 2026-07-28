"""Lookahead-bias audit for engine.run().

Test 1 (truncation reproducibility): if the engine peeks at future bars, then
re-running on data truncated at bar K would produce DIFFERENT entry prices /
stops / T1 / T2 for trades that already signalled before K. Every level is
compared exactly.

Test 2 (future-bar perturbation): scramble all bars after K to absurd values.
Any signal/level at or before K that changes is proof of a forward leak.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/selvaganapathypari/Documents/trading_tool")
sys.path.insert(0, os.path.dirname(__file__))
from engine import run  # noqa: E402

COLS = ["signal_time", "entry_time", "entry_price", "init_stop", "t1", "t2"]
SYMS = ["RELIANCE", "YESBANK", "ITC", "TATASTEEL", "INFY"]


def load(s):
    return pd.read_parquet(f"/Users/selvaganapathypari/Documents/trading_tool/investigations/data/{s}_NS_1d.parquet")


def cmp(a, b, name):
    a = a[COLS].reset_index(drop=True)
    b = b[COLS].reset_index(drop=True)
    m = min(len(a), len(b))
    if m == 0:
        print(f"  {name}: no overlapping trades to compare")
        return True
    a, b = a.iloc[:m], b.iloc[:m]
    bad = []
    for col in COLS:
        if col.endswith("time"):
            neq = (a[col].values != b[col].values)
        else:
            neq = ~np.isclose(a[col].astype(float), b[col].astype(float), rtol=1e-12, equal_nan=True)
        if neq.any():
            bad.append((col, int(neq.sum())))
    print(f"  {name}: compared {m} trades x {len(COLS)} fields -> {'MISMATCH ' + str(bad) if bad else 'identical'}")
    return not bad


ok = True
for s in SYMS:
    d = load(s)
    full = run(d, use_targets=True)["trades"]
    print(f"\n{s}: {len(full)} trades on full history")
    for frac in (0.4, 0.6, 0.8):
        K = int(len(d) * frac)
        tr = run(d.iloc[:K], use_targets=True)["trades"]
        # only compare trades whose ENTRY completed inside the truncated window
        sub = full[full["exit_time"] <= d.index[K - 1]]
        ok &= cmp(sub, tr[tr["exit_time"] <= d.index[K - 1]], f"truncate@{frac:.0%} (K={K})")

    # Test 2: destroy the future, keep the past
    K = int(len(d) * 0.6)
    pert = d.copy()
    rng = np.random.default_rng(0)
    tail = pert.iloc[K:].copy()
    mult = rng.uniform(0.3, 3.0, size=len(tail))
    for col in ["open", "high", "low", "close"]:
        tail[col] = tail[col].to_numpy() * mult
    pert.iloc[K:] = tail
    tr_p = run(pert, use_targets=True)["trades"]
    sub_full = full[full["exit_time"] <= d.index[K - 1]]
    sub_pert = tr_p[tr_p["exit_time"] <= d.index[K - 1]]
    ok &= cmp(sub_full, sub_pert, "future-scrambled@60%")

print("\nLOOKAHEAD AUDIT:", "PASS" if ok else "FAIL")
