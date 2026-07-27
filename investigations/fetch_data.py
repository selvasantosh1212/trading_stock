"""Fetch and cache daily OHLC for NSE stocks used in this investigation.

Run once: PYTHONPATH=. python investigations/fetch_data.py
Caches parquet files under investigations/data/ so we don't hammer Yahoo
on every re-run of the backtest scripts.
"""
import time
from pathlib import Path

from stock_hh_ll_tool.data import fetch_ohlc

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

SYMBOLS = [
    # --- original 5-stock sample (kept for continuity) ---
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    # Candidates for a non-bull-run test window (part 4):
    "YESBANK.NS",   # collapsed 2018-2020, then flat/declining for years
    "ONGC.NS",      # long multi-year flat/declining stretch
    # --- broader universe for this investigation (20-25+ liquid NSE names,
    # spread across sectors to avoid cherry-picking / sector concentration) ---
    # Banking / Financials
    "KOTAKBANK.NS",
    "AXISBANK.NS",
    "SBIN.NS",
    "INDUSINDBK.NS",
    # IT
    "WIPRO.NS",
    "HCLTECH.NS",
    "TECHM.NS",
    # Pharma / Healthcare
    "SUNPHARMA.NS",
    "DRREDDY.NS",
    "CIPLA.NS",
    "DIVISLAB.NS",
    # Auto
    "MARUTI.NS",
    "TVSMOTOR.NS",
    "M&M.NS",
    "BAJAJ-AUTO.NS",
    "EICHERMOT.NS",
    # FMCG / Consumer
    "HINDUNILVR.NS",
    "ITC.NS",
    "NESTLEIND.NS",
    "BRITANNIA.NS",
    "ASIANPAINT.NS",
    # Energy / Utilities
    "NTPC.NS",
    "POWERGRID.NS",
    "COALINDIA.NS",
    # Metals / Materials / Infra
    "TATASTEEL.NS",
    "ULTRACEMCO.NS",
    "LT.NS",
    # Telecom
    "BHARTIARTL.NS",
]

if __name__ == "__main__":
    failed = []
    for sym in SYMBOLS:
        fp = DATA_DIR / f"{sym.replace('.', '_')}_1d.parquet"
        if fp.exists():
            print(f"skip {sym}, already cached")
            continue
        for attempt in range(3):
            try:
                print(f"fetching {sym} (attempt {attempt + 1}) ...")
                df = fetch_ohlc(sym, "1d")
                df.to_parquet(fp)
                print(f"  {len(df)} bars, {df.index.min()} -> {df.index.max()}")
                break
            except Exception as exc:
                print(f"  failed: {exc}")
                time.sleep(5 * (attempt + 1))
        else:
            failed.append(sym)
        time.sleep(1.5)

    if failed:
        print(f"\nFAILED after retries: {failed}")
