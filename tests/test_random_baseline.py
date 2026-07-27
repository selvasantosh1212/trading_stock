import pandas as pd

from smc_validator.backtest.engine import generate_random_baseline_signals


def test_random_baseline_preserves_risk_and_target_r_multiples():
    idx = pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC")
    bars = pd.DataFrame({"close": [1.00 + 0.001 * i for i in range(20)]}, index=idx)

    real_signals = pd.DataFrame(
        {"direction": ["bearish"], "entry": [1.00], "stop": [1.02], "target1": [0.97], "target2": [0.94]},
        index=[idx[0]],
    )

    baseline = generate_random_baseline_signals(bars, real_signals, seed=1)

    assert len(baseline) == 1
    risk = abs(baseline["entry"].iloc[0] - baseline["stop"].iloc[0])
    real_risk = 0.02
    assert abs(risk - real_risk) < 1e-9

    t1_r = abs(baseline["target1"].iloc[0] - baseline["entry"].iloc[0]) / risk
    assert abs(t1_r - 1.5) < 1e-9  # (1.00-0.97)/0.02
