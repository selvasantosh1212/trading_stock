import json
from pathlib import Path

import pandas as pd

from smc_validator.data_ingestion.yahoo_chart import fetch_chart_json, parse_chart_json

RAW_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "eurusd_daily_raw.json"
PROCESSED_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "eurusd_daily.parquet"
)


def download_raw(range_: str = "25y", out_path: Path = RAW_PATH) -> Path:
    data = fetch_chart_json("EURUSD=X", range_, "1d")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data))
    return out_path


def parse_raw(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    data = json.loads(Path(raw_path).read_text())
    df = parse_chart_json(data)

    # Yahoo's last bar is often the still-open current day (intraday
    # snapshot, not a real close) — drop any bar not preceded by a full UTC
    # calendar day gap from the prior bar's date, i.e. keep only bars whose
    # date differs from "today" in the feed's own last timestamp... simplest
    # safe rule: drop the final row, since a forex day is only "closed" once
    # the NEXT day's bar exists.
    return df.iloc[:-1]


def load_daily(refresh: bool = False) -> pd.DataFrame:
    if refresh or not PROCESSED_PATH.exists():
        if refresh or not RAW_PATH.exists():
            download_raw()
        df = parse_raw()
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PROCESSED_PATH)
        return df
    return pd.read_parquet(PROCESSED_PATH)
