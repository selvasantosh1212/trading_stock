"""Daily entry point that accumulates Bank Nifty (^NSEBANK) intraday OHLC
history across 1m/5m/15m/1h. Yahoo's free chart API only exposes a limited
lookback window per interval (see banknifty_intraday.MAX_RANGE_BY_INTERVAL),
so this must run regularly (see the GitHub Actions workflow) for the
1m/5m/15m history to build up past that window over time — a single run
today cannot backfill 6 months for anything shorter than 1h.

Run as `python -m scripts.update_banknifty_data` from the repo root.
"""

from smc_validator.data_ingestion.banknifty_intraday import MAX_RANGE_BY_INTERVAL, update_history


def main() -> None:
    for interval in MAX_RANGE_BY_INTERVAL:
        df = update_history(interval)
        print(f"{interval}: {len(df)} bars, {df.index.min()} -> {df.index.max()}")


if __name__ == "__main__":
    main()
