"""Characterize the WINNING trades from the validated SMC EURUSD backtest.

Read-only reuse of scripts/run_backtest.py::build_signals and
smc_validator/backtest/engine.py::simulate_trades to regenerate the exact
151-trade sample, then enrich winners with:
  1. session at entry (Asia/London/NY + overlaps)
  2. day-of-week / hour-of-day distribution
  3. month/quarter distribution
  4. ATR(14) on 15m at entry, bucketed into terciles of trailing ~60-day ATR history

Nothing here modifies smc_validator/ or scripts/ — only imports from them.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from smc_validator.backtest.engine import simulate_trades
from smc_validator.config import load_config
from smc_validator.data_ingestion.resample import resample_ohlc
from smc_validator.liquidity.sessions import compute_session_levels
from smc_validator.patterns.order_blocks import atr as compute_atr

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from run_backtest import build_signals  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)


def classify_session(ts: pd.Timestamp) -> str:
    """Bucket a UTC timestamp into session / overlap using the rough bands
    given in the task: Asia ~00:00-09:00 UTC, London ~08:00-17:00 UTC,
    New York ~13:00-22:00 UTC."""
    h = ts.hour
    asia = 0 <= h < 9
    london = 8 <= h < 17
    ny = 13 <= h < 22
    if london and ny:
        return "London/NY overlap"
    if asia and london:
        return "Asia/London overlap"
    if london:
        return "London"
    if ny:
        return "New York"
    if asia:
        return "Asia"
    return "Off-session (22:00-00:00 UTC gap)"


def main() -> None:
    cfg = load_config()
    print("Building signals (full pipeline)...")
    bars, signals = build_signals(cfg)
    print(f"{len(signals)} candidate signals from {len(bars)} 5m bars.")

    trades = simulate_trades(bars, signals, staged_exit_split=tuple(cfg["risk"]["staged_exit_split"]))
    print(f"{len(trades)} simulated trades.")

    trades = trades.copy()
    trades["is_win"] = trades["total_r"] > 0
    print(f"Win rate: {trades['is_win'].mean():.3%} ({trades['is_win'].sum()}/{len(trades)})")

    # ---- enrichment ----
    trades["session"] = trades["signal_time"].apply(classify_session)
    trades["dow"] = trades["signal_time"].dt.day_name()
    trades["hour_utc"] = trades["signal_time"].dt.hour
    trades["month"] = trades["signal_time"].dt.to_period("M").astype(str)
    trades["quarter"] = trades["signal_time"].dt.to_period("Q").astype(str)

    # ATR(14) on 15m, aligned back to signal_time via merge_asof (last closed 15m bar)
    df_15m = resample_ohlc(bars, "15min")
    atr_15m = compute_atr(df_15m, window=14)
    atr_df = atr_15m.rename("atr15").reset_index()
    atr_df.columns = ["timestamp", "atr15"]
    atr_df = atr_df.dropna(subset=["atr15"]).sort_values("timestamp")

    trades_sorted = trades.sort_values("signal_time")
    merged = pd.merge_asof(
        trades_sorted, atr_df, left_on="signal_time", right_on="timestamp", direction="backward"
    )

    # trailing ~60-day ATR distribution terciles: for each trade, look at the
    # ATR series over the preceding 60 days (in 15m-bar terms: 60*24*4=5760 bars,
    # but we just use a time-based window) and find where the current atr15
    # value ranks within that trailing window.
    atr_series = atr_df.set_index("timestamp")["atr15"]

    def trailing_percentile(row):
        window_start = row["signal_time"] - pd.Timedelta(days=60)
        hist = atr_series.loc[window_start:row["signal_time"]]
        hist = hist.dropna()
        if len(hist) < 20:
            return np.nan
        return (hist < row["atr15"]).mean()

    merged["atr_trailing_pct"] = merged.apply(trailing_percentile, axis=1)

    def tercile(p):
        if pd.isna(p):
            return "insufficient_history"
        if p < 1 / 3:
            return "low"
        if p < 2 / 3:
            return "medium"
        return "high"

    merged["vol_regime"] = merged["atr_trailing_pct"].apply(tercile)
    merged = merged.drop(columns=["timestamp"])

    merged.to_csv(OUT_DIR / "trades_enriched.csv", index=False)
    print(f"Saved enriched trades to {OUT_DIR / 'trades_enriched.csv'}")

    winners = merged[merged["is_win"]].copy()
    losers = merged[~merged["is_win"]].copy()
    winners.to_csv(OUT_DIR / "winners_enriched.csv", index=False)
    losers.to_csv(OUT_DIR / "losers_enriched.csv", index=False)

    n_win = len(winners)
    n_all = len(merged)
    print(f"\n=== WINNERS: {n_win} / {n_all} ({n_win/n_all:.1%}) ===\n")

    # --- 1. session ---
    print("--- Session distribution (winners) ---")
    sess_win = winners["session"].value_counts(normalize=True).sort_index()
    sess_all = merged["session"].value_counts(normalize=True).sort_index()
    sess_counts = winners["session"].value_counts().sort_index()
    for s in sess_all.index:
        wc = sess_counts.get(s, 0)
        wp = sess_win.get(s, 0.0)
        ap = sess_all.get(s, 0.0)
        # win rate within that session (of ALL trades in that session)
        session_all_n = (merged["session"] == s).sum()
        session_win_n = (winners["session"] == s).sum()
        wr = session_win_n / session_all_n if session_all_n else float("nan")
        print(f"  {s:28s}: {wc:3d} winners ({wp:5.1%} of all winners) | win rate within session: {wr:.1%} ({session_win_n}/{session_all_n})")

    # --- 2. day of week / hour ---
    print("\n--- Day of week distribution (winners) ---")
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_counts = winners["dow"].value_counts().reindex(dow_order).fillna(0).astype(int)
    for d, c in dow_counts.items():
        all_n = (merged["dow"] == d).sum()
        print(f"  {d:10s}: {c:3d} winners ({c/n_win:5.1%} of winners) | win rate: {c/all_n:.1%} ({c}/{all_n})" if all_n else f"  {d}: 0")

    print("\n--- Hour of day (UTC) distribution (winners) ---")
    hour_counts = winners["hour_utc"].value_counts().sort_index()
    for h, c in hour_counts.items():
        all_n = (merged["hour_utc"] == h).sum()
        print(f"  {h:02d}:00 UTC: {c:3d} winners ({c/n_win:5.1%} of winners) | win rate: {c/all_n:.1%} ({c}/{all_n})")

    # --- 3. month/quarter ---
    print("\n--- Month distribution (winners) ---")
    month_counts = winners["month"].value_counts().sort_index()
    for m, c in month_counts.items():
        all_n = (merged["month"] == m).sum()
        print(f"  {m}: {c:3d} winners ({c/n_win:5.1%} of winners) | win rate: {c/all_n:.1%} ({c}/{all_n})")

    print("\n--- Quarter distribution (winners) ---")
    q_counts = winners["quarter"].value_counts().sort_index()
    for q, c in q_counts.items():
        all_n = (merged["quarter"] == q).sum()
        print(f"  {q}: {c:3d} winners ({c/n_win:5.1%} of winners) | win rate: {c/all_n:.1%} ({c}/{all_n})")

    # --- 4. volatility regime ---
    print("\n--- Volatility regime (trailing 60-day ATR tercile) distribution (winners) ---")
    vol_order = ["low", "medium", "high", "insufficient_history"]
    vol_counts = winners["vol_regime"].value_counts().reindex(vol_order).fillna(0).astype(int)
    for v, c in vol_counts.items():
        all_n = (merged["vol_regime"] == v).sum()
        wr = c / all_n if all_n else float("nan")
        print(f"  {v:22s}: {c:3d} winners ({c/n_win:5.1%} of winners) | win rate: {wr:.1%} ({c}/{all_n})" if all_n else f"  {v}: 0")

    print("\n--- ATR15 raw stats (absolute price units) ---")
    print(f"  Winners  mean ATR15: {winners['atr15'].mean():.6f}, median: {winners['atr15'].median():.6f}")
    print(f"  Losers   mean ATR15: {losers['atr15'].mean():.6f}, median: {losers['atr15'].median():.6f}")
    print(f"  All      mean ATR15: {merged['atr15'].mean():.6f}, median: {merged['atr15'].median():.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
