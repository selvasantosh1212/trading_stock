"""Part 2(a)/(b): direct synthetic tests for lookahead bias.

(a) Does align_htf_to_ltf really withhold an HTF bar's info until it has
    closed, i.e. does a daily bar in the MIDDLE of a still-forming week ever
    see that week's (not-yet-closed) trend?
(b) Does the entry actually execute using only information known as of the
    signal bar, at a price achievable AFTER that information was known (not
    the signal bar's own close, which in reality you can't transact at the
    instant it prints)?
"""
import numpy as np
import pandas as pd

from smc_validator.data_ingestion.resample import resample_ohlc, align_htf_to_ltf

pd.set_option("display.width", 160)

# ---- (a) synthetic weekly/daily lookahead test -----------------------------
# Build 3 full trading weeks (Mon-Fri) of daily bars. Week 2 (the middle week)
# makes a dramatic new high on its Wednesday that, if leaked early, would
# flip an "HTF trend" indicator mid-week. We use a trivial HTF series here
# (just the weekly close) rather than full compute_structure, to isolate
# align_htf_to_ltf's own behavior from the structure engine.
dates = pd.bdate_range("2024-01-01", periods=15, freq="B")  # 3 weeks x 5 bdays
assert len(dates) == 15

closes = [
    100, 101, 100, 101, 102,     # week 1 (Mon 1/1 .. Fri 1/5)
    103, 104, 999, 104, 105,     # week 2 (Mon 1/8 .. Fri 1/12) -- huge spike Wed 1/10
    106, 107, 108, 109, 110,     # week 3 (Mon 1/15 .. Fri 1/19)
]
daily = pd.DataFrame(
    {"open": closes, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes], "close": closes},
    index=dates,
)

weekly = resample_ohlc(daily, "W")
print("=== weekly bars (label = bin end per resample_ohlc's own convention) ===")
print(weekly)

# A trivial "trend" series: bullish if weekly close is a new high vs prior weekly close.
weekly_trend = pd.Series(index=weekly.index, dtype=object)
prev_close = None
for ts, row in weekly.iterrows():
    weekly_trend.loc[ts] = "spike_week" if row["close"] == 999 or row["high"] >= 999 else "normal"
    prev_close = row["close"]

aligned = align_htf_to_ltf(weekly_trend, daily.index)
result = pd.DataFrame({"daily_close": daily["close"], "aligned_htf_trend": aligned})
print("\n=== daily bars with HTF trend aligned on ===")
print(result)

wed_week2 = pd.Timestamp("2024-01-10")  # the spike day itself
mon_week2 = pd.Timestamp("2024-01-08")
mon_week3 = pd.Timestamp("2024-01-15")

leak_mid_week = result.loc[mon_week2:wed_week2, "aligned_htf_trend"].eq("spike_week").any()
sees_after_close = result.loc[mon_week3, "aligned_htf_trend"] == "spike_week"

print(f"\nLEAK CHECK: does any bar Mon 1/8 .. Wed 1/10 (before week 2 closes on Fri) already see 'spike_week'? "
      f"{'YES -- BUG' if leak_mid_week else 'no, correctly withheld'}")
print(f"CONFIRM: does Mon 1/15 (the trading day AFTER week 2 closed) correctly see 'spike_week'? "
      f"{'yes, correct' if sees_after_close else 'NO -- missing/broken propagation'}")

thu_fri_week2 = result.loc["2024-01-11":"2024-01-12", "aligned_htf_trend"]
print(f"\nThu 1/11 / Fri 1/12 (still technically inside week 2, week not yet closed) see: "
      f"\n{thu_fri_week2}")
print("(if these already show 'spike_week' that is ALSO a leak, since week 2 doesn't close until end of Fri 1/12's session)")

# ---- (b) entry execution realism -------------------------------------------
print("\n\n=== (b) Entry execution realism ===")
print(
    "compute_structure's `event`/`trend` for bar i are derived from bar i's own\n"
    "close (close_only=True). That is fine as *information* timing (you do learn\n"
    "the close once the session ends). The bug risk is in the BACKTEST, not the\n"
    "structure engine: a naive backtest that fills the trade 'at bar i's close'\n"
    "the moment event[i] fires is assuming you can transact at a price you only\n"
    "observed after the fact -- i.e. the closing print itself. That price is not\n"
    "tradeable in real time. The realistic fill is the OPEN of bar i+1, once the\n"
    "market reopens after you've had the chance to see bar i's confirmed close.\n"
    "See run_reproduce.py, which runs BOTH conventions side by side to quantify\n"
    "the gap this introduces."
)
