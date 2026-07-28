import numpy as np
import pandas as pd

from stock_hh_ll_tool.targets import find_next_target, next_liquidity_targets


def test_bullish_target_picks_nearest_of_multiple_older_unbroken_highs():
    df = pd.DataFrame(
        {
            "high": [1.30, 1.10, 1.20, 1.05, 1.06],
            "low": [1.25, 1.05, 1.15, 1.00, 1.02],
            "close": [1.28, 1.08, 1.18, 1.02, 1.04],
        }
    )
    # hand-crafted structure: swing_high_idx references bar0 (1.30) through
    # bars 1-2, then updates to bar2 (1.20) at bar3 (as if a bearish
    # reversal reset it to the more recent, lower peak), then goes to NaN at
    # bar4 (a bullish break just consumed/invalidated it) — matches real
    # compute_structure semantics without needing to drive a full price
    # sequence through it. event_direction is all None: neither 1.30 nor
    # 1.20 was ever actually broken, both remain valid "unbroken" candidates.
    structure = pd.DataFrame({
        "swing_high_idx": [np.nan, 0, 0, 2, np.nan],
        "event_direction": [None, None, None, None, None],
    })

    result = find_next_target(df, structure, "bullish")

    assert result["next_target"] == 1.20  # nearest of {1.30, 1.20} above current close 1.04
    assert result["note"] is None


def test_bullish_target_excludes_a_level_that_was_actually_broken():
    # Same shape as above, but this time bar3's swing_high_idx update to
    # bar2 (1.20) coincides with a bullish event firing right then — i.e.
    # 1.20 itself got broken through before price later retraced back down
    # to 1.04. It must NOT be offered as "next resistance" even though its
    # price sits above the current close; only 1.30 (never broken) should.
    df = pd.DataFrame(
        {
            "high": [1.30, 1.10, 1.20, 1.05, 1.06],
            "low": [1.25, 1.05, 1.15, 1.00, 1.02],
            "close": [1.28, 1.08, 1.18, 1.02, 1.04],
        }
    )
    structure = pd.DataFrame({
        "swing_high_idx": [np.nan, 0, 0, 2, np.nan],
        "event_direction": [None, None, None, "bullish", None],
    })

    result = find_next_target(df, structure, "bullish")

    assert result["next_target"] == 1.30  # 1.20 excluded: it was broken at bar3


def test_bearish_target_picks_nearest_of_multiple_older_unbroken_lows():
    df = pd.DataFrame(
        {
            "high": [1.10, 1.20, 1.15, 1.30, 1.28],
            "low": [1.00, 1.15, 0.90, 1.20, 1.22],
            "close": [1.05, 1.18, 0.95, 1.25, 1.26],
        }
    )
    # symmetric case: swing_low_idx references bar0 (low 1.00), then updates
    # to bar2 (low 0.90) at bar3, then NaN at bar4 (bearish break consumed it)
    structure = pd.DataFrame({
        "swing_low_idx": [np.nan, 0, 0, 2, np.nan],
        "event_direction": [None, None, None, None, None],
    })

    result = find_next_target(df, structure, "bearish")

    assert result["next_target"] == 1.00  # nearest of {1.00, 0.90} below current close 1.26 (1.00 is closer)


def test_no_older_level_available_reports_blue_sky_breakout():
    df = pd.DataFrame({"high": [1.00, 1.02], "low": [0.98, 1.00], "close": [0.99, 1.05]})
    structure = pd.DataFrame({"swing_high_idx": [np.nan, np.nan], "event_direction": [None, None]})

    result = find_next_target(df, structure, "bullish")

    assert result["next_target"] is None
    assert "blue-sky" in result["note"]


def _daily_df_all_same_week(closes: list[float], highs: list[float] | None = None) -> pd.DataFrame:
    # all bars land in the same ISO week/month, so previous_period_levels'
    # shift(1) has no prior period at all (NaN) -> isolates the swing-high
    # pool from the week/month levels for tests that don't care about them
    n = len(closes)
    highs = highs if highs is not None else [c + 0.5 for c in closes]
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": [c - 0.5 for c in closes], "close": closes},
        index=pd.date_range("2026-06-02", periods=n, freq="D", tz="UTC"),  # a Tuesday, same week
    )


def test_next_liquidity_targets_picks_nearest_untaken_swing_highs():
    # Two swing highs, neither exceeding the other (idx1 high=103.5, idx3
    # high=103.0 — LOWER, so it doesn't wick-sweep idx1's level), then a
    # pullback so today's close sits below both — both remain valid
    # "above current price, never swept" candidates.
    daily_df = _daily_df_all_same_week([100, 103, 101, 102.5, 98])
    daily_external_structure = pd.DataFrame({"swing_high_idx": [np.nan, 1, 1, 3, 3]})

    result = next_liquidity_targets(daily_df, daily_external_structure)

    assert result["t1"]["price"] == 103.0  # idx3's level: nearer to current price
    assert result["t1"]["label"] == "prior swing high"
    assert result["t2"]["price"] == 103.5  # idx1's level: further


def test_next_liquidity_targets_excludes_a_level_swept_by_a_wick_not_just_a_close():
    # idx0's swing high (101.5) gets WICKED through at idx2 (high=102.0)
    # even though idx2's CLOSE (100.5) never crosses it — find_next_target's
    # close-only definition would still call this "unbroken"; the stricter
    # liquidity-pool definition must not.
    daily_df = pd.DataFrame(
        {
            "open": [100.0, 100.5, 101.8, 100.0],
            "high": [101.5, 101.0, 102.0, 100.5],
            "low": [99.5, 100.0, 100.2, 99.6],
            "close": [100.0, 100.5, 100.5, 100.0],
        },
        index=pd.date_range("2026-06-02", periods=4, freq="D", tz="UTC"),
    )
    daily_external_structure = pd.DataFrame({"swing_high_idx": [np.nan, 0, 0, 0]})

    result = next_liquidity_targets(daily_df, daily_external_structure)

    assert result["t1"]["price"] is None  # the only candidate level was swept by the wick


def test_next_liquidity_targets_includes_previous_weeks_untaken_high():
    # week 1: Mon-Fri with a high of 110 on the Wednesday; week 2 (today)
    # hasn't traded through 110 yet -> previous week's high is a valid,
    # untaken candidate target
    week1 = pd.DataFrame(
        {"open": [100, 105, 108, 106, 104], "high": [101, 106, 110, 107, 105],
         "low": [99, 104, 107, 105, 103], "close": [100, 105, 109, 106, 104]},
        index=pd.date_range("2026-06-01", periods=5, freq="D", tz="UTC"),  # Mon-Fri
    )
    week2_today = pd.DataFrame(
        {"open": [103], "high": [104], "low": [101], "close": [102]},
        index=pd.date_range("2026-06-08", periods=1, freq="D", tz="UTC"),  # following Monday
    )
    daily_df = pd.concat([week1, week2_today])
    daily_external_structure = pd.DataFrame({"swing_high_idx": [np.nan] * len(daily_df)})  # isolate: no swing highs

    result = next_liquidity_targets(daily_df, daily_external_structure)

    assert result["t1"]["price"] == 110.0
    assert result["t1"]["label"] == "previous week's high"


def test_next_liquidity_targets_deduplicates_a_level_that_is_both_a_swing_high_and_the_weeks_high():
    # same price data as the previous test, but this time the Wednesday
    # bar (idx2, high=110) is ALSO a confirmed daily-external swing high —
    # the same real level shouldn't be offered twice as both T1 and T2
    # under two different labels.
    week1 = pd.DataFrame(
        {"open": [100, 105, 108, 106, 104], "high": [101, 106, 110, 107, 105],
         "low": [99, 104, 107, 105, 103], "close": [100, 105, 109, 106, 104]},
        index=pd.date_range("2026-06-01", periods=5, freq="D", tz="UTC"),
    )
    week2_today = pd.DataFrame(
        {"open": [103], "high": [104], "low": [101], "close": [102]},
        index=pd.date_range("2026-06-08", periods=1, freq="D", tz="UTC"),
    )
    daily_df = pd.concat([week1, week2_today])
    daily_external_structure = pd.DataFrame({"swing_high_idx": [np.nan, np.nan, 2, 2, 2, 2]})

    result = next_liquidity_targets(daily_df, daily_external_structure)

    assert result["t1"]["price"] == 110.0
    assert result["t1"]["label"] == "prior swing high"  # swing-high label wins the tie
    assert result["t2"]["price"] is None  # no second, distinct level available
