from stock_hh_ll_tool.position_state import load_positions, save_positions


def test_load_missing_file_returns_empty_dict(tmp_path):
    positions = load_positions(tmp_path / "does_not_exist.json")
    assert positions == {}


def test_save_then_load_roundtrips(tmp_path):
    path = tmp_path / "positions.json"
    data = {"RELIANCE.NS": {"state": "LONG", "entry_price": 1200.5, "entry_date": "2026-01-01", "stop_price": 1150.0}}

    save_positions(data, path)
    loaded = load_positions(path)

    assert loaded == data
