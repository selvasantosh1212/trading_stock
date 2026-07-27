"""Phase 1 full-checklist backtest pipeline: wires together structure,
liquidity, order-block/FVG, and signal-assembly modules on real EURUSD data,
then simulates and reports results. This is orchestration/glue specific to
this one backtest run — the reusable logic all lives in tested
smc_validator/ modules; this script just wires them together with the
concrete timeframe roles from strategy_config.yaml.

Known phase-1 scope simplifications (documented, not silent):
- Zone "active" window is a fixed timeout from the order-block bar, not a
  real mitigation tracker (we didn't build OB mitigation, only FVG
  mitigation) — a zone_active flag time-limited to `ltf_confirmation.timeout_bars`.
- Targets are exactly the video's own two staged targets (nearest
  session-liquidity level + previous day's level in the bias direction),
  not a fully generic "nearest N untaken levels" search.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from smc_validator.backtest.engine import generate_random_baseline_signals, simulate_trades
from smc_validator.backtest.metrics import compute_metrics
from smc_validator.config import load_config
from smc_validator.data_ingestion.resample import align_htf_to_ltf, resample_ohlc
from smc_validator.data_ingestion.yahoo_intraday import load_intraday
from smc_validator.liquidity.daily_weekly import previous_period_levels
from smc_validator.liquidity.sessions import compute_session_levels
from smc_validator.patterns.fvg import detect_fvg
from smc_validator.patterns.order_blocks import atr as compute_atr
from smc_validator.patterns.order_blocks import detect_order_blocks
from smc_validator.signal.checklist import assemble_signals
from smc_validator.signal.htf_bias import combine_bias
from smc_validator.signal.ltf_confirmation import detect_ltf_confirmation
from smc_validator.structure.bos_choch import compute_structure
from smc_validator.structure.swings import detect_fractal_pivots

REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "backtest_summary.md"


def _trend_series(df: pd.DataFrame, cfg: dict) -> pd.Series:
    left = cfg["structure"]["external"]["left_bars"]
    right = cfg["structure"]["external"]["right_bars"]
    pivots = detect_fractal_pivots(df, left, right)
    structure = compute_structure(df, pivots, close_only=cfg["structure"]["close_only_break"])
    return structure["trend"]


DUKASCOPY_5M_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "eurusd_5m_dukascopy.parquet"


def _load_5m_data() -> pd.DataFrame:
    # prefer the longer Dukascopy-sourced history (Phase 1.5) over the
    # 60-day-capped free Yahoo Finance window used for the initial Phase 1 run
    if DUKASCOPY_5M_PATH.exists():
        df = pd.read_parquet(DUKASCOPY_5M_PATH)
        print(f"Using Dukascopy 5m data: {len(df)} bars, {df.index[0]} to {df.index[-1]}")
        return df
    print("Dukascopy 5m data not found — falling back to Yahoo Finance's 60-day window.")
    return load_intraday("5m").iloc[:-1]


def build_signals(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_5m = _load_5m_data()
    df_1h = load_intraday("1h").iloc[:-1]
    return build_signals_from_data(df_5m, df_1h, cfg)


def build_signals_from_data(df_5m: pd.DataFrame, df_1h: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The actual signal-generation pipeline, decoupled from where the data
    came from — `build_signals` feeds it the cached Dukascopy/Yahoo
    historical data for backtesting; the live monitor (smc_validator/live/)
    feeds it freshly-fetched current data instead. Same logic either way.
    """
    # bound 1h/4H history to the same window the 5m/15m data actually covers,
    # since bias and entry timeframes must overlap for a signal to exist
    df_1h = df_1h.loc[df_5m.index[0] - pd.Timedelta(days=10) :]

    df_15m = resample_ohlc(df_5m, "15min")
    df_4h = resample_ohlc(df_1h, "4h")

    trend_1h = _trend_series(df_1h, cfg)
    trend_4h = _trend_series(df_4h, cfg)
    trend_5m = _trend_series(df_5m, cfg)

    bias_1h_on_5m = align_htf_to_ltf(trend_1h, df_5m.index)
    bias_4h_on_5m = align_htf_to_ltf(trend_4h, df_5m.index)
    htf_bias_5m = combine_bias({"4H": bias_4h_on_5m, "1H": bias_1h_on_5m})

    atr_15m = compute_atr(df_15m)
    fvg_15m = detect_fvg(df_15m, min_gap_size=cfg["fair_value_gaps"]["min_gap_size_atr_fraction"] * atr_15m.fillna(0))
    structure_15m = compute_structure(df_15m, detect_fractal_pivots(
        df_15m, cfg["structure"]["external"]["left_bars"], cfg["structure"]["external"]["right_bars"]
    ), close_only=cfg["structure"]["close_only_break"])
    ob_lookahead_bars = 5
    ob_15m = detect_order_blocks(
        df_15m,
        atr_15m,
        balance_body_atr_fraction=cfg["order_blocks"]["balance_candle_body_atr_fraction"],
        expansion_atr_multiple=cfg["order_blocks"]["expansion_move_atr_multiple"],
        lookahead=ob_lookahead_bars,
        fvg_flags=fvg_15m if cfg["order_blocks"]["require_associated_fvg"] else None,
        # tie order blocks to a genuine confirmed BOS/CHoCH, not just any
        # ATR-multiple price move — an ATR move alone fires far too often on
        # real intraday data (~1/12 bars) to be a meaningful "smart money" zone
        structure_event=structure_15m["event"],
        structure_event_direction=structure_15m["event_direction"],
    )

    # detect_order_blocks is intentionally a vectorized, OFFLINE "look at the
    # next `lookahead` bars" detector (see its docstring): ob_15m["bullish_ob"]
    # / ["bearish_ob"] at row i depend on future_max_high/future_min_low and
    # the FVG/structure-event lookahead windows, all of which reach forward
    # to bar i+lookahead-1. That flag is therefore not actually knowable in
    # real time until bar i+lookahead-1 closes — using it to start the
    # zone-active window AT bar i (as an earlier version of this script did)
    # is look-ahead bias: it lets trades between i and i+lookahead-1 react to
    # an order block that, causally, doesn't exist yet. Shifting the flags
    # (and the OB candle's own low/high, which stops/zone boxes reference)
    # forward by lookahead-1 bars makes them "arrive" on the same bar they
    # actually become knowable — this is exactly what the Pine port's
    # f_zone15 does causally already (see its header comment and the
    # idx = lookahead - 1 look-back it applies at each new bar).
    shift_n = ob_lookahead_bars - 1
    bullish_ob_15m = ob_15m["bullish_ob"].shift(shift_n).fillna(False).astype(bool)
    bearish_ob_15m = ob_15m["bearish_ob"].shift(shift_n).fillna(False).astype(bool)
    ob_low_15m = df_15m["low"].shift(shift_n)
    ob_high_15m = df_15m["high"].shift(shift_n)

    zone_direction_15m = pd.Series(pd.NA, index=df_15m.index, dtype="object")
    # bullish wins on a same-bar tie (both flags true on the same knowable
    # bar) — matches Pine's `zoneDir := bullishOb ? "bullish" : "bearish"`,
    # which checks bullishOb first. Assign bearish first so bullish's
    # assignment (if also true) overwrites it, giving the same last-write-wins
    # outcome as Pine's first-branch-wins.
    zone_direction_15m[bearish_ob_15m] = "bearish"
    zone_direction_15m[bullish_ob_15m] = "bullish"
    zone_start_15m = bullish_ob_15m | bearish_ob_15m

    timeout = cfg["ltf_confirmation"]["timeout_bars"]
    zone_active_15m = pd.Series(False, index=df_15m.index)
    zone_direction_filled_15m = pd.Series(pd.NA, index=df_15m.index, dtype="object")
    bars_since_zone_start: int | None = None
    current_direction = None
    for i in range(len(df_15m)):
        # reset on EVERY new zone_start, not just when the window has fully
        # expired — otherwise overlapping OBs blend into one giant "active"
        # block and this loop's own timeout accounting silently stops meaning
        # anything (this is exactly the bug that produced ~99% active bars
        # during development, before order blocks were tied to structure events)
        if zone_start_15m.iloc[i]:
            bars_since_zone_start = 0
            current_direction = zone_direction_15m.iloc[i]
        elif bars_since_zone_start is not None:
            bars_since_zone_start += 1
        if bars_since_zone_start is not None and bars_since_zone_start <= timeout:
            zone_active_15m.iloc[i] = True
            zone_direction_filled_15m.iloc[i] = current_direction

    zone_active_5m = align_htf_to_ltf(zone_active_15m.astype("boolean"), df_5m.index).fillna(False).astype(bool)
    zone_direction_5m = align_htf_to_ltf(zone_direction_filled_15m, df_5m.index)
    # stop is placed beyond the SAME-direction order block that produced the
    # zone the trade came from (a bullish trade's stop sits below the
    # bullish OB's low, a bearish trade's above the bearish OB's high) — was
    # previously cross-wired to the opposite-direction OB's extreme, an
    # unrelated/possibly-stale level from a different part of the chart
    ob_extreme_low_5m = align_htf_to_ltf(ob_low_15m.where(bullish_ob_15m).ffill(), df_5m.index)
    ob_extreme_high_5m = align_htf_to_ltf(ob_high_15m.where(bearish_ob_15m).ffill(), df_5m.index)

    # the real "how long to wait" timeout is already baked into zone_active_5m
    # above (in 15-min-bar units); pass a large value here so this second,
    # bar-count-since-zone-start check inside detect_ltf_confirmation (which
    # operates in 5-min bars and would otherwise double-apply a mismatched
    # unit) never becomes the binding constraint
    confirmation_5m = detect_ltf_confirmation(trend_5m, htf_bias_5m, zone_active_5m, timeout_bars=len(df_5m))

    session_low_5m = compute_session_levels(df_5m, "Asia/Tokyo", "09:00", "18:00")["session_low"]
    session_high_5m = compute_session_levels(df_5m, "Asia/Tokyo", "09:00", "18:00")["session_high"]
    prev_day_5m = previous_period_levels(df_5m, "D")

    stop_buffer = cfg["risk"]["stop_buffer_atr_fraction"]
    atr_5m_on_15m_scale = align_htf_to_ltf(atr_15m, df_5m.index)

    entry_price = df_5m["close"]
    stop_price = pd.Series(np.nan, index=df_5m.index, dtype="float64")
    target1_price = pd.Series(np.nan, index=df_5m.index, dtype="float64")
    target2_price = pd.Series(np.nan, index=df_5m.index, dtype="float64")

    bearish_mask = htf_bias_5m == "bearish"
    bullish_mask = htf_bias_5m == "bullish"
    stop_price[bearish_mask] = ob_extreme_high_5m[bearish_mask] + stop_buffer * atr_5m_on_15m_scale[bearish_mask]
    stop_price[bullish_mask] = ob_extreme_low_5m[bullish_mask] - stop_buffer * atr_5m_on_15m_scale[bullish_mask]
    target1_price[bearish_mask] = session_low_5m.shift(1)[bearish_mask]
    target1_price[bullish_mask] = session_high_5m.shift(1)[bullish_mask]
    target2_price[bearish_mask] = prev_day_5m["prev_period_low"][bearish_mask]
    target2_price[bullish_mask] = prev_day_5m["prev_period_high"][bullish_mask]

    signals = assemble_signals(
        htf_bias_5m, confirmation_5m, zone_direction_5m, entry_price, stop_price, target1_price, target2_price
    )
    signals = signals.dropna(subset=["stop", "target1", "target2"])
    return df_5m, signals


def main() -> None:
    cfg = load_config()
    bars, signals = build_signals(cfg)
    print(f"Generated {len(signals)} candidate signals from {len(bars)} 5-minute bars.")

    trades = simulate_trades(bars, signals, staged_exit_split=tuple(cfg["risk"]["staged_exit_split"]))
    metrics = compute_metrics(trades)

    lines = [
        "# Full-Checklist Backtest Summary (Phase 1, preliminary)",
        "",
        f"Instrument: {cfg['instrument']}. Bars: {len(bars)} x 5-minute "
        f"({bars.index[0]} to {bars.index[-1]}).",
        f"Candidate signals: {len(signals)}. Simulated trades: {len(trades)}.",
        "",
    ]
    if metrics.get("n_trades", 0) > 0:
        lines += [
            "## Full checklist (real signals)",
            f"- Win rate: {metrics['win_rate']:.1%}",
            f"- Expectancy: {metrics['expectancy_r']:.2f} R",
            f"- Avg win / avg loss: {metrics['avg_win_r']:.2f}R / {metrics['avg_loss_r']:.2f}R",
            f"- Profit factor: {metrics['profit_factor']:.2f}",
            f"- Max drawdown: {metrics['max_drawdown_r']:.2f} R",
            f"- Trades/month: {metrics['trades_per_month']:.1f}",
            "",
        ]

        baseline_runs = []
        for seed in [1, 2, 3, 4, 5]:
            baseline_signals = generate_random_baseline_signals(bars, signals, seed=seed)
            baseline_trades = simulate_trades(bars, baseline_signals, staged_exit_split=tuple(cfg["risk"]["staged_exit_split"]))
            baseline_runs.append(compute_metrics(baseline_trades))

        avg_baseline_expectancy = sum(m["expectancy_r"] for m in baseline_runs if m.get("n_trades", 0) > 0) / len(
            [m for m in baseline_runs if m.get("n_trades", 0) > 0]
        )
        avg_baseline_win_rate = sum(m["win_rate"] for m in baseline_runs if m.get("n_trades", 0) > 0) / len(
            [m for m in baseline_runs if m.get("n_trades", 0) > 0]
        )
        lines += [
            "## Random-entry baseline (same risk/target sizing, 5 seeds averaged)",
            f"- Avg win rate: {avg_baseline_win_rate:.1%}",
            f"- Avg expectancy: {avg_baseline_expectancy:.2f} R",
            "",
            "## Verdict",
            (
                f"Real checklist expectancy ({metrics['expectancy_r']:.2f}R) vs random-entry baseline "
                f"({avg_baseline_expectancy:.2f}R) with identical stop/target sizing: "
                + (
                    "checklist outperforms the random baseline."
                    if metrics["expectancy_r"] > avg_baseline_expectancy
                    else "checklist does NOT outperform the random baseline — the specific timing/location "
                    "logic is not adding value beyond what similarly-sized stops/targets get by chance in this sample."
                )
            ),
            "",
            f"Sample size caveat: only {metrics['n_trades']} trades over ~"
            f"{(bars.index[-1] - bars.index[0]).days} days of data ({bars.index[0].date()} to "
            f"{bars.index[-1].date()}) — still a modest sample for drawing a confident conclusion either "
            "way. This is a preliminary directional read, not a final verdict.",
        ]
    else:
        lines.append("No trades were simulated — see notes on candidate signal count above.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
