import lzma
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
POINT_VALUE = {"EURUSD": 100_000.0}
TICK_STRUCT = struct.Struct(">iiiff")  # ms_offset, ask_raw, bid_raw, ask_vol, bid_vol


def _hour_url(symbol: str, hour: pd.Timestamp) -> str:
    # Dukascopy's path uses a 0-indexed month, unlike pandas/calendar convention
    return f"{BASE_URL}/{symbol}/{hour.year}/{hour.month - 1:02d}/{hour.day:02d}/{hour.hour:02d}h_ticks.bi5"


def _fetch_hour_ticks(
    symbol: str, hour: pd.Timestamp, point_value: float, session: requests.Session, max_retries: int = 5
) -> pd.DataFrame:
    # Dukascopy's public endpoint rate-limits aggressively with no
    # Retry-After header — back off exponentially on 429 rather than
    # hammering it (this is what turned a 3-day pull into an instant, silent
    # all-empty result the first time this was tried). Transient connection
    # timeouts/errors are just as real over a multi-hour sustained pull (this
    # crashed an entire 12-month download on one bad hour originally) — treat
    # them the same way: retry with backoff, only give up on this one hour
    # after exhausting retries, never let one hour's failure kill the batch.
    #
    # `giveup_reason` distinguishes "retries exhausted" (429s or connection
    # errors — worth logging, since it means real data may be missing from
    # the final result) from "got a clean response that just wasn't 200/had
    # no content" (e.g. a genuinely quiet weekend hour) — silently returning
    # empty for both used to hide the former's failure trail entirely.
    backoff = 3.0
    resp = None
    giveup_reason = None
    for attempt in range(max_retries):
        try:
            resp = session.get(_hour_url(symbol, hour), timeout=15)
        except requests.exceptions.RequestException as e:
            resp = None
            giveup_reason = f"connection error ({e.__class__.__name__})"
        else:
            if resp.status_code != 429:
                giveup_reason = None
                break
            giveup_reason = "rate limited (429)"
        time.sleep(backoff)
        backoff *= 2

    if giveup_reason is not None:
        print(f"  WARNING: giving up on {symbol} {hour} after {max_retries} attempts — {giveup_reason}")

    if resp is None or resp.status_code != 200 or len(resp.content) == 0:
        return pd.DataFrame(columns=["timestamp", "bid", "ask"])

    try:
        decompressed = lzma.decompress(resp.content, format=lzma.FORMAT_AUTO)
    except lzma.LZMAError:
        return pd.DataFrame(columns=["timestamp", "bid", "ask"])

    n = len(decompressed) // TICK_STRUCT.size
    if n == 0:
        return pd.DataFrame(columns=["timestamp", "bid", "ask"])

    rows = [TICK_STRUCT.unpack_from(decompressed, i * TICK_STRUCT.size) for i in range(n)]
    ms, ask_raw, bid_raw, _ask_vol, _bid_vol = zip(*rows)
    timestamps = hour + pd.to_timedelta(ms, unit="ms")
    return pd.DataFrame(
        {"timestamp": timestamps, "bid": [b / point_value for b in bid_raw], "ask": [a / point_value for a in ask_raw]}
    )


def _is_likely_trading_hour(hour: pd.Timestamp) -> bool:
    # forex market is closed roughly Fri 21:00 UTC to Sun 21:00 UTC — skip
    # those hours to avoid wasting requests on files that are always empty
    dow, hh = hour.dayofweek, hour.hour
    if dow == 5:  # Saturday
        return False
    if dow == 6 and hh < 21:  # Sunday before ~21:00 UTC
        return False
    if dow == 4 and hh >= 21:  # Friday after ~21:00 UTC
        return False
    return True


def fetch_range_ohlc(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    bar_freq: str = "5min",
    max_workers: int = 4,
    progress_every: int = 200,
) -> pd.DataFrame:
    """Downloads Dukascopy tick data hour-by-hour over [start, end) and
    aggregates each hour directly to OHLC bars on the MID price
    ((bid+ask)/2) — avoids holding months of raw ticks in memory. Skips
    hours outside typical forex trading hours to cut wasted requests.
    """
    point_value = POINT_VALUE[symbol]
    hours = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    hours = [h for h in hours if _is_likely_trading_hour(h)]

    all_bars = []
    session = requests.Session()
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_hour_ticks, symbol, h, point_value, session): h for h in hours}
        for future in as_completed(futures):
            try:
                ticks = future.result()
            except Exception as e:
                # belt-and-suspenders: _fetch_hour_ticks already retries
                # transient errors internally, but a single hour's
                # unretryable failure should never take down a multi-hour
                # batch download — skip it and keep going.
                print(f"  WARNING: failed to fetch {futures[future]}: {e}")
                done += 1
                continue
            done += 1
            if done % progress_every == 0:
                print(f"  fetched {done}/{len(hours)} hours...")
            if len(ticks) == 0:
                continue
            ticks = ticks.set_index("timestamp").sort_index()
            mid = (ticks["bid"] + ticks["ask"]) / 2
            bars = mid.resample(bar_freq, label="right", closed="left").ohlc()
            all_bars.append(bars.dropna(how="any"))

    if not all_bars:
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    combined = pd.concat(all_bars).sort_index()
    return combined[~combined.index.duplicated(keep="first")]


def download_and_cache(
    symbol: str, start: pd.Timestamp, end: pd.Timestamp, bar_freq: str, out_path: Path, **kwargs
) -> pd.DataFrame:
    df = fetch_range_ohlc(symbol, start, end, bar_freq=bar_freq, **kwargs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    return df
