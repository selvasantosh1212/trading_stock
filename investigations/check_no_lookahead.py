"""Part 4: Standalone lookahead sanity checks for the investigation's
backtest engine (investigations/strategy.py). Does NOT modify smc_validator
or stock_hh_ll_tool - only checks how this investigation's own code (which
consumes those tested modules) behaves.

Checks performed:
  1. Every generated trade fills strictly AFTER its own signal bar (entry
     fill index > entry signal index; exit fill index > exit signal index) -
     enforced already by `assert` inside generate_trades(), re-verified here
     from the returned DataFrame's dates against the source index.
  2. Truncation test: re-running the backtest on a prefix of the data (all
     bars up to and including some cutoff date) must reproduce IDENTICAL
     trades up to that cutoff as the full-history run. If future bars could
     silently influence a past signal, truncating the future would change
     past signals - this is the standard walk-forward lookahead detector.
  3. HTF alignment check: for a sample of daily bars, confirm the aligned
     HTF trend value visible on date D never comes from an HTF (weekly) bar
     whose own timestamp is AFTER D (i.e. align_htf_to_ltf never leaks a
     still-forming weekly bar backward).

Run: PYTHONPATH=. python investigations/check_no_lookahead.py
"""
from pathlib import Path

import pandas as pd

from investigations.strategy import DEFAULT_PARAMS, build_structures, generate_trades

DATA_DIR = Path(__file__).parent / "data"


def check_fill_after_signal(symbol: str = "RELIANCE.NS"):
    df = pd.read_parquet(DATA_DIR / f"{symbol.replace('.', '_')}_1d.parquet")
    htf_struct, ltf_struct, htf_trend_on_ltf = build_structures(df, DEFAULT_PARAMS)
    trades, open_trade = generate_trades(df, htf_trend_on_ltf, ltf_struct)

    idx = df.index
    for _, t in trades.iterrows():
        entry_signal_pos = idx.get_loc(t["entry_signal_date"])
        entry_fill_pos = idx.get_loc(t["entry_date"])
        assert entry_fill_pos == entry_signal_pos + 1, "entry fill is not the very next bar after signal"
        assert entry_fill_pos > entry_signal_pos

        exit_signal_pos = idx.get_loc(t["exit_signal_date"])
        exit_fill_pos = idx.get_loc(t["exit_date"])
        assert exit_fill_pos == exit_signal_pos + 1, "exit fill is not the very next bar after signal"
        assert exit_fill_pos > exit_signal_pos
        # entry price must be that fill bar's OPEN, not its close/high/low
        assert abs(t["entry_price"] - df["open"].iloc[entry_fill_pos]) < 1e-9
        assert abs(t["exit_price"] - df["open"].iloc[exit_fill_pos]) < 1e-9

    print(f"[OK] {symbol}: {len(trades)} trades, all fills strictly one bar after their signal, at that bar's OPEN.")


def check_truncation_invariance(symbol: str = "RELIANCE.NS", cutoff_frac: float = 0.7):
    df_full = pd.read_parquet(DATA_DIR / f"{symbol.replace('.', '_')}_1d.parquet")
    cutoff = int(len(df_full) * cutoff_frac)
    cutoff_date = df_full.index[cutoff]

    df_trunc = df_full.iloc[: cutoff + 1]

    htf_s_full, ltf_s_full, trend_full = build_structures(df_full, DEFAULT_PARAMS)
    trades_full, _ = generate_trades(df_full, trend_full, ltf_s_full)

    htf_s_trunc, ltf_s_trunc, trend_trunc = build_structures(df_trunc, DEFAULT_PARAMS)
    trades_trunc, open_trunc = generate_trades(df_trunc, trend_trunc, ltf_s_trunc)

    # Compare trades whose ENTRY happened comfortably before the cutoff
    # (giving a few bars of margin since a pivot straddling the cutoff can
    # legitimately be interpreted differently near the edge of either
    # dataset - that's an edge effect, not lookahead. We check trades whose
    # entry_signal_date is at least 15 bars before the cutoff.).
    margin_date = df_full.index[cutoff - 15]

    full_before = trades_full[trades_full["entry_signal_date"] <= margin_date].reset_index(drop=True)
    trunc_before = trades_trunc[trades_trunc["entry_signal_date"] <= margin_date].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        full_before[["entry_signal_date", "entry_date", "entry_price"]],
        trunc_before[["entry_signal_date", "entry_date", "entry_price"]],
        check_dtype=False,
    )
    print(
        f"[OK] {symbol}: {len(full_before)} trades entered well before the {cutoff_frac:.0%} cutoff "
        f"({cutoff_date.date()}) are IDENTICAL whether computed on the full 10y history or on data "
        f"truncated at the cutoff. No future data is leaking into past signals."
    )


def check_htf_alignment_no_future_leak(symbol: str = "RELIANCE.NS"):
    df = pd.read_parquet(DATA_DIR / f"{symbol.replace('.', '_')}_1d.parquet")
    from smc_validator.data_ingestion.resample import resample_ohlc

    weekly = resample_ohlc(df, "W")
    htf_struct, ltf_struct, htf_trend_on_ltf = build_structures(df, DEFAULT_PARAMS)

    # For every daily bar, the merge_asof-aligned HTF trend must come from a
    # weekly bar whose own timestamp (=its close time, since resample uses
    # label="right") is <= the daily bar's own timestamp.
    weekly_trend = htf_struct["trend"]
    weekly_times = weekly.index
    violations = 0
    for daily_ts in df.index[::37]:  # sample every 37th bar for speed
        aligned_val = htf_trend_on_ltf.loc[daily_ts]
        if pd.isna(aligned_val):
            continue
        eligible = weekly_times[weekly_times <= daily_ts]
        if len(eligible) == 0:
            violations += 1
            continue
        expected_val = weekly_trend.loc[eligible[-1]]
        if expected_val != aligned_val:
            violations += 1

    assert violations == 0, f"{violations} sampled bars saw an HTF trend from a not-yet-closed weekly bar"
    print(f"[OK] {symbol}: sampled daily bars only ever see HTF trend from an already-closed weekly bar.")


if __name__ == "__main__":
    for sym in ["RELIANCE.NS", "TCS.NS", "YESBANK.NS"]:
        check_fill_after_signal(sym)
        check_truncation_invariance(sym)
        check_htf_alignment_no_future_leak(sym)
    print("\nAll lookahead sanity checks passed.")
