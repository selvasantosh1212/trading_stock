import pandas as pd

from smc_validator.data_ingestion.resample import resample_ohlc
from smc_validator.data_ingestion.yahoo_chart import fetch_chart_json, parse_chart_json

# Yahoo's free chart API caps how far back intraday intervals can go — same
# limits we hit with forex. Irrelevant for daily-bar swing signals; for the
# intraday confirmation layer, 60 days is plenty since this is a LIVE
# screener checking "what's true right now", not a multi-year backtest.
MAX_RANGE_BY_INTERVAL = {
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "1h": "730d",
    "4h": "730d",  # not a native Yahoo interval — fetched as 1h and resampled below
    "1d": "10y",
}


def fetch_ohlc(symbol: str, interval: str) -> pd.DataFrame:
    """Fetches OHLC bars for any Yahoo Finance symbol (NSE stocks use the
    ticker + '.NS' suffix, e.g. 'TCS.NS'). No historical caching — this tool
    always wants the freshest available bar, unlike the forex validation
    work which needed long, stable history for backtesting.
    """
    fetch_interval = "1h" if interval == "4h" else interval
    data = fetch_chart_json(symbol, MAX_RANGE_BY_INTERVAL[interval], fetch_interval)
    df = parse_chart_json(data)

    if interval == "4h":
        df = resample_ohlc(df, "4h")

    return df
