"""Core engine for the Weekly-HTF / Daily-LTF structure-following swing
strategy under investigation. Read-only reuse of smc_validator's tested
structure/resample primitives - nothing here modifies those modules.

Strategy recap:
  - Entry: HTF (weekly) trend is 'bullish' (i.e. HTF already confirmed a
    bullish BOS/CHoCH) AND the LTF's (daily) OWN structure fires a bullish
    CHoCH on this bar (the daily pullback just reversed back up).
  - Exit: hold until the HTF (weekly) trend flips to 'bearish'. By
    construction of compute_structure, the bar that flips a trend from
    bullish to bearish is always tagged CHoCH (bearish) - so "aligned HTF
    trend turns bearish" IS "HTF confirms a bearish CHoCH".
  - No stop-loss, no profit target - pure trend-state following.

Lookahead safety:
  - Weekly bars are timestamped at their own close (resample_ohlc uses
    label="right"), and compute_structure's trend/event at a bar is knowable
    exactly at that bar's own close (it never touches a pivot before that
    pivot's `confirmed_at` bar - enforced inside compute_structure itself).
  - align_htf_to_ltf does a backward asof merge: a daily bar only ever sees
    an HTF value whose timestamp is <= its own timestamp, so a still-forming
    weekly bar is never leaked into an earlier daily bar.
  - Both the entry and exit CONDITIONS are therefore knowable using bar i's
    own close. This engine never fills a trade on bar i itself - it always
    fills at bar i+1's OPEN, strictly after the signal bar (see the
    `assert fill_i > i` lines in generate_trades, and check_no_lookahead.py
    for a standalone sanity script).
"""
from dataclasses import dataclass

import pandas as pd

from smc_validator.data_ingestion.resample import align_htf_to_ltf, resample_ohlc
from smc_validator.structure.bos_choch import compute_structure
from smc_validator.structure.swings import detect_fractal_pivots


@dataclass
class PivotParams:
    htf_left: int
    htf_right: int
    ltf_left: int
    ltf_right: int

    @property
    def label(self) -> str:
        return f"htf{self.htf_left}-{self.htf_right}_ltf{self.ltf_left}-{self.ltf_right}"


DEFAULT_PARAMS = PivotParams(htf_left=5, htf_right=5, ltf_left=5, ltf_right=5)


def build_structures(daily_df: pd.DataFrame, params: PivotParams = DEFAULT_PARAMS, htf_freq: str = "W"):
    """Returns (htf_structure_on_htf_index, ltf_structure_on_daily_index,
    htf_trend_aligned_to_daily_index)."""
    htf_df = resample_ohlc(daily_df, htf_freq)
    htf_pivots = detect_fractal_pivots(htf_df, params.htf_left, params.htf_right)
    htf_structure = compute_structure(htf_df, htf_pivots, close_only=True)

    ltf_pivots = detect_fractal_pivots(daily_df, params.ltf_left, params.ltf_right)
    ltf_structure = compute_structure(daily_df, ltf_pivots, close_only=True)

    htf_trend_on_ltf = align_htf_to_ltf(htf_structure["trend"], daily_df.index)

    return htf_structure, ltf_structure, htf_trend_on_ltf


def generate_trades(daily_df: pd.DataFrame, htf_trend_on_ltf: pd.Series, ltf_structure: pd.DataFrame):
    """Event-driven long-only state machine. Fills strictly on the bar AFTER
    the signal bar (next bar's open) - never the signal bar's own close.
    Returns (closed_trades_df, open_trade_dict_or_None).
    """
    n = len(daily_df)
    idx = daily_df.index
    open_ = daily_df["open"].to_numpy()
    close = daily_df["close"].to_numpy()

    htf_trend = htf_trend_on_ltf.to_numpy()
    ltf_event = ltf_structure["event"].to_numpy()
    ltf_dir = ltf_structure["event_direction"].to_numpy()

    trades = []
    in_position = False
    entry_signal_i = entry_fill_i = None
    entry_price = None

    for i in range(n):
        if not in_position:
            if htf_trend[i] == "bullish" and ltf_event[i] == "CHoCH" and ltf_dir[i] == "bullish":
                fill_i = i + 1
                if fill_i >= n:
                    continue  # signal on final bar, no bar left to fill on - drop
                assert fill_i > i, "lookahead violation: fill must be strictly after signal bar"
                in_position = True
                entry_signal_i = i
                entry_fill_i = fill_i
                entry_price = open_[fill_i]
        else:
            if htf_trend[i] == "bearish":
                fill_i = i + 1
                if fill_i >= n:
                    continue  # exit signal on final bar - leave trade open, handled below
                assert fill_i > i, "lookahead violation: fill must be strictly after signal bar"
                exit_price = open_[fill_i]
                trades.append(
                    {
                        "entry_signal_date": idx[entry_signal_i],
                        "entry_date": idx[entry_fill_i],
                        "entry_price": entry_price,
                        "exit_signal_date": idx[i],
                        "exit_date": idx[fill_i],
                        "exit_price": exit_price,
                        "return_pct": (exit_price / entry_price - 1.0) * 100.0,
                        "holding_days": fill_i - entry_fill_i,
                        "still_open": False,
                    }
                )
                in_position = False
                entry_signal_i = entry_fill_i = entry_price = None

    trades_df = pd.DataFrame(trades)

    open_trade = None
    if in_position:
        # mark-to-market at last available close, clearly flagged as unrealized
        open_trade = {
            "entry_signal_date": idx[entry_signal_i],
            "entry_date": idx[entry_fill_i],
            "entry_price": entry_price,
            "exit_signal_date": pd.NaT,
            "exit_date": idx[-1],
            "exit_price": close[-1],
            "return_pct": (close[-1] / entry_price - 1.0) * 100.0,
            "holding_days": (n - 1) - entry_fill_i,
            "still_open": True,
        }

    return trades_df, open_trade


def build_equity_curve(daily_df: pd.DataFrame, trades_df: pd.DataFrame, open_trade: dict | None) -> pd.Series:
    """Flat-cash-between-trades compounded equity curve on the full daily
    index, for drawdown / time-in-market measurement. Starts at 1.0. No
    interest accrues while flat (matches "no position" reality).
    """
    idx = daily_df.index
    close = daily_df["close"]
    equity = pd.Series(1.0, index=idx)

    all_trades = list(trades_df.to_dict("records"))
    if open_trade is not None:
        all_trades.append(open_trade)

    level = 1.0
    prev_exit_date = idx[0]
    for t in all_trades:
        entry_date, exit_date = t["entry_date"], t["exit_date"]
        # flat from prev exit to this entry
        equity.loc[prev_exit_date:entry_date] = level
        seg = close.loc[entry_date:exit_date]
        if len(seg) > 0:
            equity.loc[entry_date:exit_date] = level * (seg / seg.iloc[0])
            level = float(equity.loc[exit_date])
        prev_exit_date = exit_date
    equity.loc[prev_exit_date:] = level

    return equity


def buy_and_hold_curve(daily_df: pd.DataFrame) -> pd.Series:
    close = daily_df["close"]
    return close / close.iloc[0]


def max_drawdown_pct(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min() * 100.0)


def time_in_market_pct(daily_df: pd.DataFrame, trades_df: pd.DataFrame, open_trade: dict | None) -> float:
    idx = daily_df.index
    in_mkt = pd.Series(False, index=idx)
    all_trades = list(trades_df.to_dict("records"))
    if open_trade is not None:
        all_trades.append(open_trade)
    for t in all_trades:
        in_mkt.loc[t["entry_date"] : t["exit_date"]] = True
    return float(in_mkt.mean() * 100.0)


# ---------------------------------------------------------------------------
# Compatibility layer used by run_reproduce.py / run_stoploss.py /
# run_subperiod.py (this investigation's original scripts). Kept alongside
# the PivotParams-based API above so both remain usable. Uses the brief's
# originally-specified pivot params: HTF weekly 5/5, LTF daily 3/3 (NOT the
# 5/5 DEFAULT_PARAMS above, which is a different parameterization choice).
# Adds stop-loss support (structure / fixed-pct), which the class-based API
# above does not have, for part 3 of the investigation.
# ---------------------------------------------------------------------------
from dataclasses import dataclass as _dataclass  # noqa: E402

import numpy as _np  # noqa: E402


@_dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    stop_price: float | None = None

    @property
    def ret_pct(self) -> float:
        return (self.exit_price / self.entry_price - 1.0) * 100.0


def run_backtest(
    daily: pd.DataFrame,
    htf_freq: str = "W",
    entry_fill: str = "next_open",
    stop_mode: str = "none",
    stop_pct: float = 0.06,
) -> list[Trade]:
    """See module docstring / run_reproduce.py for the fill-convention
    rationale. HTF pivots use left=right=5, LTF pivots use left=right=3,
    matching the brief's reference logic exactly."""
    weekly = resample_ohlc(daily, htf_freq)
    htf_pivots = detect_fractal_pivots(weekly, 5, 5)
    htf_struct = compute_structure(weekly, htf_pivots, close_only=True)

    ltf_pivots = detect_fractal_pivots(daily, 3, 3)
    ltf_struct = compute_structure(daily, ltf_pivots, close_only=True)

    htf_trend_on_ltf = align_htf_to_ltf(htf_struct["trend"], daily.index)

    entry_ok = (
        (htf_trend_on_ltf == "bullish")
        & (ltf_struct["event"] == "CHoCH")
        & (ltf_struct["event_direction"] == "bullish")
    ).to_numpy()
    exit_ok = (htf_trend_on_ltf == "bearish").to_numpy()

    close = daily["close"].to_numpy()
    open_ = daily["open"].to_numpy()
    low = daily["low"].to_numpy()
    swing_low = ltf_struct["swing_low"].to_numpy()
    idx = daily.index
    n = len(daily)

    trades: list[Trade] = []
    in_pos = False
    entry_price = entry_time = stop_price = entry_bar = None

    for i in range(n):
        if not in_pos:
            if entry_ok[i]:
                if entry_fill == "signal_close":
                    fill_i, fill_price = i, close[i]
                else:
                    if i + 1 >= n:
                        continue
                    fill_i, fill_price = i + 1, open_[i + 1]

                in_pos = True
                entry_price = fill_price
                entry_time = idx[fill_i]
                entry_bar = fill_i

                if stop_mode == "structure":
                    stop_price = swing_low[i] if not _np.isnan(swing_low[i]) else None
                elif stop_mode == "pct":
                    stop_price = entry_price * (1 - stop_pct)
                else:
                    stop_price = None
        else:
            if i > entry_bar:
                if stop_price is not None and low[i] <= stop_price:
                    trades.append(Trade(entry_time, entry_price, idx[i], stop_price, "stop", stop_price))
                    in_pos = False
                    entry_price = entry_time = stop_price = entry_bar = None
                    continue
                if exit_ok[i]:
                    if entry_fill == "signal_close":
                        fill_i, fill_price = i, close[i]
                    else:
                        if i + 1 >= n:
                            continue
                        fill_i, fill_price = i + 1, open_[i + 1]
                    trades.append(Trade(entry_time, entry_price, idx[fill_i], fill_price, "htf_choch", stop_price))
                    in_pos = False
                    entry_price = entry_time = stop_price = entry_bar = None

    return trades


def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["entry_time", "entry_price", "exit_time", "exit_price", "exit_reason", "ret_pct"])
    return pd.DataFrame(
        {
            "entry_time": [t.entry_time for t in trades],
            "entry_price": [t.entry_price for t in trades],
            "exit_time": [t.exit_time for t in trades],
            "exit_price": [t.exit_price for t in trades],
            "exit_reason": [t.exit_reason for t in trades],
            "ret_pct": [t.ret_pct for t in trades],
        }
    )


def summarize(trades_df: pd.DataFrame) -> dict:
    if len(trades_df) == 0:
        return {"n_trades": 0}
    wins = trades_df["ret_pct"] > 0
    equity = (1 + trades_df["ret_pct"] / 100.0).cumprod()
    running_max = equity.cummax()
    dd = (equity / running_max - 1.0)
    return {
        "n_trades": len(trades_df),
        "win_rate": float(wins.mean()) * 100,
        "avg_ret_pct": float(trades_df["ret_pct"].mean()),
        "median_ret_pct": float(trades_df["ret_pct"].median()),
        "worst_trade_pct": float(trades_df["ret_pct"].min()),
        "best_trade_pct": float(trades_df["ret_pct"].max()),
        "total_compounded_pct": float((equity.iloc[-1] - 1) * 100),
        "max_drawdown_pct": float(dd.min() * 100),
    }
