import pandas as pd

from smc_validator.backtest.engine import simulate_trades
from smc_validator.backtest.metrics import compute_metrics


def _bars(highs, lows):
    idx = pd.date_range("2024-01-01", periods=len(highs), freq="h", tz="UTC")
    return pd.DataFrame({"open": highs, "high": highs, "low": lows, "close": highs}, index=idx)


def test_full_win_both_targets_hit():
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    highs = [1.00, 1.00, 1.00, 0.97, 0.94, 0.94]  # bearish move down
    lows = [1.00, 1.00, 1.00, 0.97, 0.94, 0.94]
    bars = pd.DataFrame({"high": highs, "low": lows}, index=idx)

    signals = pd.DataFrame(
        {"direction": ["bearish"], "entry": [1.00], "stop": [1.02], "target1": [0.97], "target2": [0.94]},
        index=[idx[0]],
    )

    trades = simulate_trades(bars, signals)
    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["leg1_r"] == 1.5  # (1.00-0.97)/(1.02-1.00)
    assert row["leg2_r"] == 3.0  # (1.00-0.94)/(1.02-1.00)
    assert row["exit_reason"] == "both_targets"


def test_stop_before_target1_loses_full_risk():
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    bars = pd.DataFrame({"high": [1.00, 1.03, 1.03, 1.03], "low": [1.00, 1.01, 1.01, 1.01]}, index=idx)

    signals = pd.DataFrame(
        {"direction": ["bearish"], "entry": [1.00], "stop": [1.02], "target1": [0.97], "target2": [0.94]},
        index=[idx[0]],
    )

    trades = simulate_trades(bars, signals)
    assert len(trades) == 1
    assert trades.iloc[0]["total_r"] == -1.0
    assert trades.iloc[0]["exit_reason"] == "stop_before_target1"


def test_stop_after_partial_only_loses_second_leg():
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    # bar1: hits target1 (0.97); bar2: reverses back up through stop (1.02)
    bars = pd.DataFrame({"high": [1.00, 1.00, 1.03, 1.03], "low": [1.00, 0.96, 1.01, 1.01]}, index=idx)

    signals = pd.DataFrame(
        {"direction": ["bearish"], "entry": [1.00], "stop": [1.02], "target1": [0.97], "target2": [0.94]},
        index=[idx[0]],
    )

    trades = simulate_trades(bars, signals)
    row = trades.iloc[0]
    assert row["leg1_r"] > 0
    assert row["leg2_r"] == -1.0
    assert row["exit_reason"] == "stop_after_partial"
    # total is the 50/50 blend, not a full loss despite leg2 losing
    assert row["total_r"] == 0.5 * row["leg1_r"] + 0.5 * (-1.0)


def test_compute_metrics_basic_shape():
    trades = pd.DataFrame(
        {
            "signal_time": pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC"),
            "total_r": [2.0, -1.0, 3.0, -1.0],
        }
    )
    metrics = compute_metrics(trades)

    assert metrics["n_trades"] == 4
    assert metrics["win_rate"] == 0.5
    assert metrics["expectancy_r"] == 0.75
    assert metrics["profit_factor"] == 2.5  # (2+3) / (1+1)
