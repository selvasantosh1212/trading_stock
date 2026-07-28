import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/selvaganapathypari/Documents/trading_tool")
sys.path.insert(0, os.path.dirname(__file__))

from engine import max_dd, run  # noqa: E402

DATA = "/Users/selvaganapathypari/Documents/trading_tool/investigations/data/*.parquet"


def load_all():
    out = {}
    for f in sorted(glob.glob(DATA)):
        sym = os.path.basename(f).replace("_NS_1d.parquet", "")
        out[sym] = pd.read_parquet(f)
    return out


def evaluate(label, data, **kw):
    all_tr, rows = [], []
    for sym, d in data.items():
        r = run(d, **kw)
        tr = r["trades"]
        if len(tr):
            tr = tr.assign(symbol=sym)
            all_tr.append(tr)
        eq = r["equity"]
        bh = d["close"] / d["close"].iloc[0]
        rows.append(
            {
                "symbol": sym,
                "n": len(tr),
                "strat_total_pct": (eq.iloc[-1] - 1) * 100,
                "bh_total_pct": (bh.iloc[-1] - 1) * 100,
                "strat_dd": max_dd(eq),
                "bh_dd": max_dd(bh),
            }
        )
    trades = pd.concat(all_tr, ignore_index=True) if all_tr else pd.DataFrame()
    per = pd.DataFrame(rows)
    wins = trades["total_r"] > 0 if len(trades) else pd.Series(dtype=bool)
    gw = trades.loc[wins, "total_r"].sum() if len(trades) else 0.0
    gl = -trades.loc[~wins, "total_r"].sum() if len(trades) else 0.0
    eqr = trades["total_r"].cumsum() if len(trades) else pd.Series([0.0])
    s = {
        "label": label,
        "n_trades": len(trades),
        "win_rate": float(wins.mean() * 100) if len(trades) else np.nan,
        "expectancy_r": float(trades["total_r"].mean()) if len(trades) else np.nan,
        "median_r": float(trades["total_r"].median()) if len(trades) else np.nan,
        "avg_win_r": float(trades.loc[wins, "total_r"].mean()) if wins.any() else 0.0,
        "avg_loss_r": float(trades.loc[~wins, "total_r"].mean()) if (~wins).any() else 0.0,
        "profit_factor": float(gw / gl) if gl > 0 else float("inf"),
        "max_dd_r": float((eqr - eqr.cummax()).min()),
        "avg_hold_days": float(trades["holding_days"].mean()) if len(trades) else np.nan,
        "med_hold_days": float(trades["holding_days"].median()) if len(trades) else np.nan,
        "avg_ret_pct": float(trades["net_return_pct"].mean()) if len(trades) else np.nan,
        "med_ret_pct": float(trades["net_return_pct"].median()) if len(trades) else np.nan,
        "worst_ret_pct": float(trades["net_return_pct"].min()) if len(trades) else np.nan,
        "strat_avg_total_pct": per["strat_total_pct"].mean(),
        "bh_avg_total_pct": per["bh_total_pct"].mean(),
        "beat_bh": int((per["strat_total_pct"] > per["bh_total_pct"]).sum()),
        "strat_avg_dd": per["strat_dd"].mean(),
        "bh_avg_dd": per["bh_dd"].mean(),
    }
    return s, trades, per


if __name__ == "__main__":
    data = load_all()
    which = sys.argv[1] if len(sys.argv) > 1 else "main"
    results = []

    if which == "main":
        cfgs = [
            ("A: deployed baseline (no target)", dict(use_targets=False)),
            ("B: NEW staged T1/T2 liquidity", dict(use_targets=True)),
            ("C: T1/T2 no breakeven move", dict(use_targets=True, breakeven_after_t1=False)),
            ("D: full exit at T1 (no runner)", dict(use_targets=True, t1_fraction=1.0)),
            ("E: daily internal/external", dict(use_targets=True, mode="daily_internal_external",
                                                htf_left=8, htf_right=8, ltf_left=2, ltf_right=2)),
        ]
    elif which == "sens":
        cfgs = [
            ("B base (5/5 W, 3/3 D, atr0.1)", dict(use_targets=True)),
            ("ltf 2/2", dict(use_targets=True, ltf_left=2, ltf_right=2)),
            ("ltf 4/4", dict(use_targets=True, ltf_left=4, ltf_right=4)),
            ("ltf 5/5", dict(use_targets=True, ltf_left=5, ltf_right=5)),
            ("htf 4/4", dict(use_targets=True, htf_left=4, htf_right=4)),
            ("htf 6/6", dict(use_targets=True, htf_left=6, htf_right=6)),
            ("atr buf 0.08 (-20%)", dict(use_targets=True, stop_atr_frac=0.08)),
            ("atr buf 0.12 (+20%)", dict(use_targets=True, stop_atr_frac=0.12)),
            ("atr buf 0.5", dict(use_targets=True, stop_atr_frac=0.5)),
            ("t1_fraction 0.33", dict(use_targets=True, t1_fraction=1 / 3)),
            ("t1_fraction 0.67", dict(use_targets=True, t1_fraction=2 / 3)),
            ("min 1R to T1", dict(use_targets=True, min_r_to_t1=1.0)),
        ]
    elif which == "rescue":
        cfgs = [
            ("A: deployed baseline", dict(use_targets=False)),
            ("B: targets as-designed", dict(use_targets=True)),
            ("min 1R to T1", dict(use_targets=True, min_r_to_t1=1.0)),
            ("min 1.5R to T1", dict(use_targets=True, min_r_to_t1=1.5)),
            ("min 2R to T1", dict(use_targets=True, min_r_to_t1=2.0)),
            ("min 3R to T1", dict(use_targets=True, min_r_to_t1=3.0)),
            ("min 2R + no BE move", dict(use_targets=True, min_r_to_t1=2.0, breakeven_after_t1=False)),
            ("min 2R + 1/3 off at T1", dict(use_targets=True, min_r_to_t1=2.0, t1_fraction=1 / 3)),
        ]
    elif which == "realism":
        cfgs = [
            ("B stop-first (pessimistic)", dict(use_targets=True, same_bar_tie="stop_first")),
            ("B target-first (optimistic)", dict(use_targets=True, same_bar_tie="target_first")),
            ("B trend exit at signal close", dict(use_targets=True, trend_exit_fill="signal_close")),
            ("B entry bar inactive", dict(use_targets=True, entry_bar_active=False)),
            ("A base stop-first", dict(use_targets=False)),
            ("A target-first", dict(use_targets=False, same_bar_tie="target_first")),
        ]
    else:
        raise SystemExit("unknown")

    rows = []
    store = {}
    for label, kw in cfgs:
        s, tr, per = evaluate(label, data, **kw)
        rows.append(s)
        store[label] = (tr, per)
        print(f"done {label}: {s['n_trades']} trades", flush=True)

    df = pd.DataFrame(rows).set_index("label")
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print(df.round(3).T.to_string())

    outdir = os.path.dirname(__file__)
    df.to_csv(os.path.join(outdir, f"summary_{which}.csv"))
    if which == "main":
        for lbl in ["B: NEW staged T1/T2 liquidity", "A: deployed baseline (no target)"]:
            tr, per = store[lbl]
            tr.to_csv(os.path.join(outdir, f"trades_{lbl[0]}.csv"), index=False)
            per.to_csv(os.path.join(outdir, f"perstock_{lbl[0]}.csv"), index=False)
        tr = store["B: NEW staged T1/T2 liquidity"][0]
        print("\n--- exit reason mix (B) ---")
        print(tr["exit_reason"].value_counts())
        print("\n--- T1/T2 distance in R (B) ---")
        print(tr[["t1_dist_r", "t2_dist_r"]].describe().round(3).to_string())
        print("T1 hit rate:", round(tr["t1_hit"].mean() * 100, 1), "%")
        print("trades with no T1 (blue sky):", int(tr["t1"].isna().sum()))
        print("trades with no T2:", int(tr["t2"].isna().sum()))
