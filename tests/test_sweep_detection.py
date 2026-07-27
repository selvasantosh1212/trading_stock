import pandas as pd

from smc_validator.liquidity.sweep_detection import detect_sweep, first_sweep_then_reverse


def test_detect_sweep_high():
    df = pd.DataFrame({"high": [1.00, 1.02, 1.06], "low": [0.99, 1.00, 1.01]})
    level = pd.Series(1.05, index=df.index)

    swept = detect_sweep(df, level, "high")
    assert list(swept) == [False, False, True]


def test_detect_sweep_low():
    df = pd.DataFrame({"high": [1.00, 1.02, 1.06], "low": [0.99, 0.90, 1.01]})
    level = pd.Series(0.95, index=df.index)

    swept = detect_sweep(df, level, "low")
    assert list(swept) == [False, True, False]


def test_first_sweep_then_reverse_requires_correct_order():
    # Asia high swept at bar1, Asia low swept at bar3 -> the "sweep high,
    # reverse, take low" pattern should fire at bar3, not before
    df = pd.DataFrame(
        {
            "high": [1.00, 1.06, 1.03, 1.01],
            "low": [0.99, 1.00, 0.98, 0.90],
        }
    )
    asia_high = pd.Series(1.05, index=df.index)
    asia_low = pd.Series(0.95, index=df.index)

    pattern = first_sweep_then_reverse(df, asia_high, "high", asia_low, "low")
    assert list(pattern) == [False, False, False, True]


def test_first_sweep_then_reverse_does_not_fire_if_order_reversed():
    # low gets swept first (bar1), high swept later (bar2) -> should NOT
    # count as "high swept then low swept"
    df = pd.DataFrame(
        {
            "high": [1.00, 1.01, 1.06, 1.01],
            "low": [0.99, 0.90, 0.98, 0.97],
        }
    )
    asia_high = pd.Series(1.05, index=df.index)
    asia_low = pd.Series(0.95, index=df.index)

    pattern = first_sweep_then_reverse(df, asia_high, "high", asia_low, "low")
    assert not pattern.any()
