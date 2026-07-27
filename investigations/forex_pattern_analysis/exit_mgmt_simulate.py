"""LOCAL COPY of smc_validator.backtest.engine.simulate_trades, instrumented
to additionally capture exit-bar timestamps (and the target1-hit timestamp)
so we can study HOW winning trades resolve over time. Read-only reuse of the
original logic -- smc_validator/ and scripts/ themselves are NOT modified.

This must stay bit-for-bit identical to the original decision logic (same-bar
stop-before-target conservatism, no lookahead, etc.) -- only the recorded
fields are extended.
"""

import pandas as pd


def simulate_trades_instrumented(
    bars: pd.DataFrame, signals: pd.DataFrame, staged_exit_split: tuple[float, float] = (0.5, 0.5)
) -> pd.DataFrame:
    bar_index = bars.index
    positions = bar_index.get_indexer(signals.index)
    results = []

    for sig_idx, pos in zip(signals.index, positions):
        if pos == -1 or pos + 1 >= len(bars):
            continue
        row = signals.loc[sig_idx]
        direction, entry, stop, t1, t2 = row["direction"], row["entry"], row["stop"], row["target1"], row["target2"]

        risk = abs(entry - stop)
        if risk == 0:
            continue

        leg1_r = None
        leg2_r = None
        exit_reason = None
        leg1_done = False
        t1_hit_time = None
        t1_hit_bars = None
        exit_time = None
        exit_bars = None

        for j in range(pos + 1, len(bars)):
            bar = bars.iloc[j]
            if direction == "bullish":
                hit_stop, hit_t1, hit_t2 = bar["low"] <= stop, bar["high"] >= t1, bar["high"] >= t2
            else:
                hit_stop, hit_t1, hit_t2 = bar["high"] >= stop, bar["low"] <= t1, bar["low"] <= t2

            if not leg1_done:
                if hit_stop:
                    leg1_r = leg2_r = -1.0
                    exit_reason = "stop_before_target1"
                    exit_time = bar_index[j]
                    exit_bars = j - pos
                    break
                if hit_t1:
                    leg1_r = abs(t1 - entry) / risk
                    leg1_done = True
                    t1_hit_time = bar_index[j]
                    t1_hit_bars = j - pos
                    continue
            else:
                if hit_stop:
                    leg2_r = -1.0
                    exit_reason = "stop_after_partial"
                    exit_time = bar_index[j]
                    exit_bars = j - pos
                    break
                if hit_t2:
                    leg2_r = abs(t2 - entry) / risk
                    exit_reason = "both_targets"
                    exit_time = bar_index[j]
                    exit_bars = j - pos
                    break
        else:
            exit_reason = "unresolved_end_of_data"

        if leg1_r is None:
            continue

        total_r = staged_exit_split[0] * leg1_r + staged_exit_split[1] * (leg2_r if leg2_r is not None else leg1_r)
        results.append(
            {
                "signal_time": sig_idx,
                "direction": direction,
                "leg1_r": leg1_r,
                "leg2_r": leg2_r,
                "total_r": total_r,
                "exit_reason": exit_reason,
                "t1_hit_time": t1_hit_time,
                "t1_hit_bars": t1_hit_bars,
                "exit_time": exit_time,
                "exit_bars": exit_bars,
            }
        )

    return pd.DataFrame(results)
