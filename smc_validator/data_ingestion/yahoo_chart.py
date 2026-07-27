import pandas as pd
import requests

CHART_URL_TEMPLATE = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


def fetch_chart_json(symbol: str, range_: str, interval: str, timeout: int = 20) -> dict:
    """Raw Yahoo Finance chart API call, shared by every symbol/interval
    fetcher in this project (forex daily/intraday, and any stock via
    stock_hh_ll_tool) — same endpoint shape, only the symbol/range/interval
    differ.
    """
    resp = requests.get(
        CHART_URL_TEMPLATE.format(symbol=symbol),
        params={"range": range_, "interval": interval},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def parse_chart_json(data: dict) -> pd.DataFrame:
    """Extracts OHLC from a Yahoo chart API JSON payload (or an equivalent
    dict loaded from a cached raw file) into a UTC-indexed DataFrame.
    """
    result = data["chart"]["result"]
    if not result:
        raise ValueError(f"No chart data in response: {data['chart'].get('error')}")
    result = result[0]
    ts = result["timestamp"]
    quote = result["indicators"]["quote"][0]

    df = pd.DataFrame(
        {"open": quote["open"], "high": quote["high"], "low": quote["low"], "close": quote["close"]},
        index=pd.to_datetime(ts, unit="s", utc=True),
    )
    df.index.name = "timestamp"
    return df.dropna(how="any").sort_index()
