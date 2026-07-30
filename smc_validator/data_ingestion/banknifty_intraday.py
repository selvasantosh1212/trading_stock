"""Bank Nifty (^NSEBANK) intraday OHLC via Yahoo's free chart API.

Yahoo caps how far back each interval can go: 1m to the last 7 days, 5m/15m
to 60 days, 1h to ~2 years. For 1h that's enough to backfill 6 months in a
single call; for the shorter intervals there's no way to get true history
today, so `update_history` instead MERGES each fetch into whatever was
already saved, letting the on-disk history accumulate day over day as this
is run repeatedly (see scripts/update_banknifty_data.py and the daily
GitHub Actions workflow that calls it).
"""

from pathlib import Path

import pandas as pd

from smc_validator.data_ingestion.yahoo_chart import fetch_chart_json, parse_chart_json

SYMBOL = "^NSEBANK"
PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

MAX_RANGE_BY_INTERVAL = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "730d"}


def _processed_path(interval: str) -> Path:
    return PROCESSED_DIR / f"banknifty_{interval}.parquet"


def fetch_latest(interval: str, timeout: int = 30) -> pd.DataFrame:
    """One fetch of whatever window Yahoo allows for this interval."""
    data = fetch_chart_json(SYMBOL, MAX_RANGE_BY_INTERVAL[interval], interval, timeout=timeout)
    return parse_chart_json(data)


def update_history(interval: str) -> pd.DataFrame:
    """Fetches the latest available window and merges it into the saved
    history for this interval, deduplicating on timestamp. Existing bars
    that have since aged out of Yahoo's window are kept rather than
    dropped, since merging (not overwriting) is what lets 1m/5m/15m history
    grow past Yahoo's lookback cap over repeated runs.
    """
    processed_path = _processed_path(interval)
    new_df = fetch_latest(interval)

    if processed_path.exists():
        existing_df = pd.read_parquet(processed_path)
        merged = pd.concat([existing_df, new_df])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = new_df

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(processed_path)
    return merged


def load_history(interval: str) -> pd.DataFrame:
    return pd.read_parquet(_processed_path(interval))
