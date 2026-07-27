import sys
from pathlib import Path

import pandas as pd

from smc_validator.data_ingestion.dukascopy import fetch_range_ohlc

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def main() -> None:
    months = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    end = pd.Timestamp(pd.Timestamp.now(tz="UTC").date(), tz="UTC")
    start = end - pd.DateOffset(months=months)

    print(f"Fetching EURUSD 5-minute bars from {start} to {end} ({months} months)...")
    df = fetch_range_ohlc("EURUSD", start, end, bar_freq="5min", max_workers=4, progress_every=100)
    print(f"Got {len(df)} bars.")

    out_path = OUT_DIR / "eurusd_5m_dukascopy.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
