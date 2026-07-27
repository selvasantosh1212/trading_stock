import numpy as np
import pandas as pd


def simulate_trades(bars: pd.DataFrame, signals: pd.DataFrame, staged_exit_split: tuple[float, float] = (0.5, 0.5)) -> pd.DataFrame:
    """Walks forward bar-by-bar after each signal's own bar, until the stop
    or both staged targets resolve. `bars` must be the same timeframe the
    signal's entry/stop/targets were computed on. No lookahead: a signal at
    position `pos` only ever looks at bars strictly after `pos`.

    Per strategy_config (`risk.breakeven_after_partial: false`), the stop
    stays at its ORIGINAL level after target1 is hit — never silently moved
    to breakeven. Same-bar stop+target ambiguity is resolved conservatively
    (assume the stop hit first), so this never overstates performance.
    """
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
                    break
                if hit_t1:
                    leg1_r = abs(t1 - entry) / risk
                    leg1_done = True
                    continue
            else:
                if hit_stop:
                    leg2_r = -1.0
                    exit_reason = "stop_after_partial"
                    break
                if hit_t2:
                    leg2_r = abs(t2 - entry) / risk
                    exit_reason = "both_targets"
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
            }
        )

    return pd.DataFrame(results)


def generate_random_baseline_signals(bars: pd.DataFrame, real_signals: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Random-entry baseline (per Phase 1 plan: "compare against trivial
    baselines through the same engine"). Reuses the REAL signals' own risk
    distance and target R-multiples — so stop/target sizing is identical —
    but places each at a uniformly random bar and random direction. Isolates
    whether the checklist's specific timing/location logic adds value beyond
    just having similarly-sized stops/targets in this market regime.
    """
    rng = np.random.default_rng(seed)
    n = len(real_signals)
    if n == 0:
        return pd.DataFrame(columns=["direction", "entry", "stop", "target1", "target2"])

    valid_positions = np.arange(len(bars) - 2)
    replace = n > len(valid_positions)
    random_positions = rng.choice(valid_positions, size=n, replace=replace)
    directions = rng.choice(["bullish", "bearish"], size=n)

    risk = (real_signals["entry"] - real_signals["stop"]).abs().to_numpy()
    t1_r = (real_signals["target1"] - real_signals["entry"]).abs().to_numpy() / risk
    t2_r = (real_signals["target2"] - real_signals["entry"]).abs().to_numpy() / risk

    rows = []
    for i, pos in enumerate(random_positions):
        entry = bars["close"].iloc[pos]
        direction = directions[i]
        r = risk[i]
        sign = 1 if direction == "bullish" else -1
        rows.append(
            {
                "direction": direction,
                "entry": entry,
                "stop": entry - sign * r,
                "target1": entry + sign * t1_r[i] * r,
                "target2": entry + sign * t2_r[i] * r,
            }
        )

    return pd.DataFrame(rows, index=bars.index[random_positions])
