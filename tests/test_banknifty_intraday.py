import pandas as pd

from smc_validator.data_ingestion import banknifty_intraday


def _bars(rows: dict[str, tuple[float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(
        rows, orient="index", columns=["open", "high", "low", "close"]
    ).set_axis(pd.to_datetime(list(rows.keys()), utc=True))


def test_update_history_merges_new_fetch_into_existing_file(tmp_path, monkeypatch):
    # Simulates Yahoo's lookback cap aging older bars out of a later fetch:
    # 00:00 only exists in what was previously saved, 00:10 is refetched
    # with a revised close, and 00:15 is brand new.
    monkeypatch.setattr(banknifty_intraday, "PROCESSED_DIR", tmp_path)
    existing = _bars(
        {
            "2024-01-01 00:00": (100.0, 101.0, 99.0, 100.5),
            "2024-01-01 00:10": (100.5, 101.5, 100.0, 101.0),
        }
    )
    existing.to_parquet(banknifty_intraday._processed_path("5m"))

    refetched = _bars(
        {
            "2024-01-01 00:10": (100.5, 102.0, 100.0, 101.8),
            "2024-01-01 00:15": (101.8, 102.5, 101.5, 102.0),
        }
    )
    monkeypatch.setattr(banknifty_intraday, "fetch_latest", lambda interval, timeout=30: refetched)

    merged = banknifty_intraday.update_history("5m")

    assert list(merged.index) == [
        pd.Timestamp("2024-01-01 00:00", tz="UTC"),
        pd.Timestamp("2024-01-01 00:10", tz="UTC"),
        pd.Timestamp("2024-01-01 00:15", tz="UTC"),
    ]
    # the newer fetch's value for the overlapping bar wins
    assert merged.loc["2024-01-01 00:10", "close"] == 101.8
    # and it's persisted, not just returned in-memory
    assert banknifty_intraday.load_history("5m").equals(merged)


def test_update_history_first_run_has_no_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(banknifty_intraday, "PROCESSED_DIR", tmp_path)
    fetched = _bars({"2024-01-01 00:00": (100.0, 101.0, 99.0, 100.5)})
    monkeypatch.setattr(banknifty_intraday, "fetch_latest", lambda interval, timeout=30: fetched)

    merged = banknifty_intraday.update_history("5m")

    assert merged.equals(fetched)
    assert banknifty_intraday.load_history("5m").equals(fetched)
