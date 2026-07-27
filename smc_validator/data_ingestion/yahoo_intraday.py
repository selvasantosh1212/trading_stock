import json
from pathlib import Path

import pandas as pd

from smc_validator.data_ingestion.yahoo_chart import fetch_chart_json, parse_chart_json

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

# Yahoo's free chart API caps how far back each interval can go — 5m/15m are
# limited to the last 60 days; 1h can go back ~2 years. This bounds the
# full-checklist backtest to however far back the SHORTEST of the timeframes
# actually used goes (see Phase 1 plan's data-scope note).
MAX_RANGE_BY_INTERVAL = {"5m": "60d", "15m": "60d", "1h": "730d"}


def download_raw(interval: str, out_path: Path | None = None) -> Path:
    data = fetch_chart_json("EURUSD=X", MAX_RANGE_BY_INTERVAL[interval], interval, timeout=30)
    out_path = out_path or RAW_DIR / f"eurusd_{interval}_raw.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data))
    return out_path


def parse_raw(raw_path: Path) -> pd.DataFrame:
    data = json.loads(Path(raw_path).read_text())
    return parse_chart_json(data)


def load_intraday(interval: str, refresh: bool = False) -> pd.DataFrame:
    processed_path = PROCESSED_DIR / f"eurusd_{interval}.parquet"
    raw_path = RAW_DIR / f"eurusd_{interval}_raw.json"

    if refresh or not processed_path.exists():
        if refresh or not raw_path.exists():
            download_raw(interval, raw_path)
        df = parse_raw(raw_path)
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(processed_path)
        return df
    return pd.read_parquet(processed_path)
