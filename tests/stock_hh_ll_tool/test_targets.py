import numpy as np
import pandas as pd

from stock_hh_ll_tool.targets import find_next_target


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
