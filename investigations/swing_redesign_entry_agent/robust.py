"""Parameter sensitivity, per-stock robustness, chronological split, and an
explicit lookahead-bias verification (prefix-truncation reproducibility)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/selvaganapathypari/Documents/trading_tool")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from decompose import simulate, structure_of, summarize  # noqa: E402
from smc_validator.data_ingestion.resample import align_htf_to_ltf, resample_ohlc  # noqa: E402

DATA = ROOT / "investigations" / "data"


def build(d):
    wk = resample_ohlc(d, "W")
    return align_htf_to_ltf(structure_of(wk, 5, 5)["trend"], d.index).to_numpy()


def agree(wtr, etr):
    return np.array(["bullish" if (x == "bullish" and y == "bullish") else "o" for x, y in zip(wtr, etr)], dtype=object)


if __name__ == "__main__":
    files = sorted(DATA.glob("*.parquet"))
    grids, perstock = {}, {}
    for f in files:
        d = pd.read_parquet(f).dropna().sort_index()
        wtr = build(d)
        for ext in (4, 5, 6, 8):
            e = structure_of(d, ext, ext)["trend"].to_numpy()
            for it in (1, 2, 3, 4):
                tg = structure_of(d, it, it)
                for tag, et in (("LIT", e), ("HYB", agree(wtr, e))):
                    t = simulate(d, et, e, tg, True, True, None)
                    if len(t):
                        t["symbol"] = f.stem
                        grids.setdefault((tag, ext, it), []).append(t)
        # per-stock for the two headline designs + existing baseline
        i2 = structure_of(d, 2, 2)
        i3 = structure_of(d, 3, 3)
        e5 = structure_of(d, 5, 5)["trend"].to_numpy()
        for tag, t in (("EXISTING", simulate(d, wtr, wtr, i3, False, False, None)),
                       ("LITERAL", simulate(d, e5, e5, i2, True, True, None)),
                       ("HYBRID", simulate(d, agree(wtr, e5), e5, i2, True, True, None))):
            if len(t):
                t["symbol"] = f.stem
                perstock.setdefault(tag, []).append(t)

    print("=== PARAMETER GRID (external NxN daily trend, internal MxM trigger) ===")
    rows = {}
    for k, v in sorted(grids.items()):
        rows[f"{k[0]} ext{k[1]} int{k[2]}"] = summarize(pd.concat(v))
    g = pd.DataFrame(rows).T
    pd.set_option("display.width", 300, "display.max_columns", 60)
    print(g[["n", "win_rate", "exp_r", "pf", "avg_hold", "med_hold", "R_per_100d", "tr_per_mo_35"]].to_string())

    print("\n=== PER-STOCK ROBUSTNESS + CHRONOLOGICAL SPLIT ===")
    for tag, v in perstock.items():
        p = pd.concat(v)
        per = p.groupby("symbol")["total_r"].agg(["count", "mean", "sum"])
        mid = p["signal_time"].min() + (p["signal_time"].max() - p["signal_time"].min()) / 2
        h1, h2 = p[p.signal_time <= mid], p[p.signal_time > mid]
        print(f"{tag:9s} stocks_with_pos_expectancy={int((per['mean']>0).sum())}/{len(per)} "
              f"| sum_R={p.total_r.sum():.1f} | 1stHalf exp_r={h1.total_r.mean():.3f}(n={len(h1)}) "
              f"2ndHalf exp_r={h2.total_r.mean():.3f}(n={len(h2)})")
        p.to_csv(Path(__file__).parent / f"final_{tag}.csv", index=False)

    print("\n=== LOOKAHEAD CHECK: truncate history, re-run, compare overlapping trades ===")
    for tag in ("EXISTING", "LITERAL", "HYBRID"):
        mism = same = 0
        for f in files[:10]:
            d = pd.read_parquet(f).dropna().sort_index()
            cut = int(len(d) * 0.6)
            for dd, label in ((d, "full"), (d.iloc[:cut], "trunc")):
                wtr = build(dd)
                e5 = structure_of(dd, 5, 5)["trend"].to_numpy()
                i2, i3 = structure_of(dd, 2, 2), structure_of(dd, 3, 3)
                t = {"EXISTING": simulate(dd, wtr, wtr, i3, False, False, None),
                     "LITERAL": simulate(dd, e5, e5, i2, True, True, None),
                     "HYBRID": simulate(dd, agree(wtr, e5), e5, i2, True, True, None)}[tag]
                if label == "full":
                    full = t
                else:
                    trunc = t
            if len(trunc) == 0:
                continue
            # compare all truncated trades that closed well before the cut date
            cutoff = d.index[cut - 1] - pd.Timedelta(days=40)
            tr = trunc[trunc.exit_date < cutoff]
            fl = full[full.exit_date < cutoff]
            if len(tr) == len(fl) and np.allclose(tr.total_r.to_numpy(), fl.total_r.to_numpy()):
                same += 1
            else:
                mism += 1
                print(f"   MISMATCH {tag} {f.stem}: trunc={len(tr)} full={len(fl)}")
        print(f"{tag:9s}: {same} stocks reproduce identically, {mism} mismatch")
