# Project Plan: Turning a YouTube Trading Strategy into a Real Tool

## The idea

We watched a YouTube video ("I Finally Revealed My Entire Trading Strategy (For Free)" by Lewis Kelly / Prosperity School) that teaches a "Smart Money Concepts" (SMC) trading strategy, and the goal is to eventually turn it into a TradingView tool. Before writing any TradingView code, we decided to first check — using real historical price data — whether the rules taught in the video actually work, rather than just trusting the video. That's what this first phase of work is about.

## The strategy, in plain English

The video teaches a step-by-step way to decide when to buy or sell a currency pair (it uses EUR/USD as the example):

1. **Figure out the trend** — look at the bigger picture (4-hour and 1-hour charts) and decide if price is generally going up or down. Only trade in that direction.
2. **Pick a target** — before entering any trade, figure out where price is likely to go. The video's two go-to targets are: (a) the high/low of the previous trading session (Asian, London, or New York), and (b) yesterday's high or low.
3. **Find a good spot to get in** — look for a specific kind of price zone (called an "order block") where big buyers/sellers were previously active, ideally paired with a small gap in price ("fair value gap") that tends to get revisited.
4. **Wait for confirmation** — drop down to a smaller timeframe (15-minute or 5-minute chart) and wait for that smaller chart to also flip into agreement with the bigger trend, before actually entering.
5. **Use a wide stop-loss** — not a tight one, because tight stops get "wicked out" (price briefly spikes through them before reversing).
6. **Take profit in stages** — close part of the trade at the nearer target, let the rest run to the further target.

One specific, checkable claim the video makes: *"After a red day that closes below the previous day's low, there's a 75% chance the next day also trades down to touch that same low."*

## What we actually built

We wrote a small Python toolkit (not the final TradingView tool yet — this is the "does it actually work" testing stage) that can:
- Read real historical EUR/USD price data
- Detect trend direction the same way the video describes (including the tricky "don't get faked out by small wiggles" rule)
- Mark out previous session/day highs and lows
- Find "order block" and "fair value gap" zones automatically
- Put all of the above together into a single "should I enter a trade right now" signal, exactly following the video's 6-step checklist
- Simulate how those trades would have actually played out, and measure the results properly (not just win rate, but whether it would have made or lost money overall)

Every piece was tested against hand-built examples with known right answers before we trusted it on real data (36 automated checks, all passing).

## What we found

**Good news — one specific claim checks out, and is even better than advertised.**
We tested the "75% chance price comes back to yesterday's low" claim on 22 years of real EUR/USD data. It held up: in the most recent, previously-unseen chunk of data, price came back to that level 80–85% of the time — clearly higher than the ~49% chance it would happen on any random day. This part of the strategy is solid.

**Mixed news at first, then more encouraging once we got more data.**
When we first ran the *entire* checklist (trend + target + zone + confirmation, all together) on real recent data, it only generated 24 trades over about 3 months, and those trades lost money on average — but so did random, similarly-sized trades in the same period. So on that small sample, the specific "smart money" logic wasn't clearly better than guessing.

The free data source we started with only gave about 60 days of detailed (5-minute) price history, so we went and got a longer, cleaner dataset directly from a tick-data provider (Dukascopy) — first about 4 months instead of 2. Re-running the exact same test on that bigger sample (50 trades instead of 24) flipped the result: 74% win rate, positive expectancy, and a clear improvement over random entries with the same-sized stops/targets.

**Then we went further and got a full 12 months of data — and the result held up, in fact got stronger.** With 151 trades (roughly 6x the original sample) over a full year: **70.9% win rate, +0.54R average return per trade, and winners now bigger than losers on average** — a clear, consistent gap over random entries with identical stop/target sizing (which sat at essentially zero on the same data). The important thing here isn't just that the number is good — it's that it went from -0.17R (24 trades) to +0.19R (50 trades) to +0.54R (151 trades) as we added more data. A result that *improves* as the sample grows is a much better sign than one that was strong early and faded — that pattern would suggest we got lucky with a small sample, and this is the opposite.

**Update, found during the Pine Script port**: line-by-line comparison against `scripts/run_backtest.py` while porting this logic to Pine surfaced a real bug — stop-loss placement was cross-wired to the *opposite*-direction order block's extreme (a bearish trade's stop referenced the most recent bullish order block's high, an unrelated/possibly-stale zone, instead of the bearish zone that actually produced the trade). This was a **Python-only fix** — the Pine script never had this bug (it always used the same-direction zone's own low/high by construction). Re-running with the fix **improved** every headline number: 153 trades, 75.8% win rate, +0.67R average return, profit factor 3.35→6.00, max drawdown -7.77R→-3.44R. A bug fix that helps results is reassuring rather than suspicious — see `reports/phase1_summary.md` Section 4 for the full reasoning.

**Second update, found during an independent audit of the above fix**: the 153-trade/75.8% numbers just above were themselves still contaminated by a separate, more serious bug — look-ahead bias in how `run_backtest.py` timed the order-block "zone." `detect_order_blocks()` is a vectorized, offline detector that looks *forward* up to `lookahead` (5) bars to decide whether bar `i` is an order block (it needs to see the expansion move, the FVG, and the confirming BOS/CHoCH, none of which exist yet at bar `i`) — so `ob_15m["bullish_ob"].iloc[i]` isn't actually knowable until bar `i+lookahead-1` closes. `run_backtest.py` was starting the zone's "active" timeout window at bar `i` itself instead of `i+lookahead-1`, letting trades react to an order block that, causally, didn't exist yet at that point in time. The Pine port (`f_zone15` in `pine_scripts/smc_zone_entry_dashboard.pine`) never had this bug — it was already written causally (it waits for the lookahead window to close, then looks back with `idx = lookahead - 1`, exactly as its own header comment describes), so **no Pine change was needed**, only a Python fix (shifting the order-block flags and the OB candle's own low/high forward by `lookahead-1` bars before they drive the zone-active window and the stop price). Re-running with this second fix changed the numbers again, this time **not uniformly for the better**: fewer, more selective trades — **75 trades, 80.0% win rate, +0.44R average return**, profit factor 3.80, max drawdown -2.49R, avg win/loss 0.74R/-0.78R, ~6.3 trades/month. Win rate went up but expectancy and profit factor came down — consistent with a chunk of the earlier 153 trades having been signals that, causally, shouldn't have fired at all. Treat **75 trades / 80.0% win rate / +0.44R / PF 3.80 / MDD -2.49R** as the current, best-effort-causal numbers; the 151/70.9%/+0.54R and 153/75.8%/+0.67R figures above are both now superseded, kept only for the historical trend of how the numbers evolved as bugs were found and fixed.

We're still keeping this in perspective: it's one instrument (EUR/USD) over one year, and a few of the strategy's fine details are still educated guesses rather than pinned down. But the evidence has crossed from "encouraging but inconclusive" into **"a real, strengthening signal worth taking seriously"** — enough to start seriously planning the TradingView (Pine Script) version, ideally alongside a couple of quick sanity checks (a second currency pair, and testing whether small tweaks to the guessed parameters change the result much) rather than treating this one result as the final word.

### What separates the winning trades from the losing ones
We took this a step further and asked: of the trades, what actually distinguishes the winners from the losers? Four parallel investigations (three on winners, one on losers, all using the same measurement definitions so they're directly comparable) looked at this from several angles — full detail in `reports/forex_trade_pattern_analysis.md`. **Re-run against the current 75-trade dataset** (this section previously quoted the original 151-trade numbers, which are now superseded — one finding below reversed outright in the process):

- **A filter that looked tempting on win rate alone, but the profit math argues against it**: how close the first profit target sits relative to the stop-loss. Nearby-target trades still win somewhat more often on their own (81.4% vs 78.1% — a much smaller gap than the original 75.2%/59.5% split), but the far-target trades still carry the majority of total profit (59.8%, down from 89% before). Lesson stands even though the effect weakened: **win rate alone can be misleading** — always check total return too.
- **Reversed — the Friday filter no longer holds.** The original run found Friday was the clear worst day (46.9% win rate); on the corrected data, Friday is now *above* the overall baseline (81.2% win rate) and no longer distinguishable from any other weekday. Skipping Friday alone now very slightly *hurts* results.
- **What actually works now, verified on total profit**: skip trades only when volatility (15-minute ATR) is in the bottom third of its recent range — no day-of-week filter. This single check keeps 81% of trades, raises win rate from 80.0% to 83.6%, and keeps 96% of the total profit. Optionally also requiring the New York or London/New York-overlap session pushes win rate to 95.5% but gives up about 35% of total profit for extra selectivity on a much smaller sample (n=22) — a more expensive trade-off than it looked before.
- **Checked and still ruled out**: the size of the "fair value gap" showed no real winner/loser difference. Order-block candle tightness now shows a small gap, but on only 8 losing trades with that data available — read as noise, not a real effect, pending more data.
- **A secondary signal that flipped direction**: confirmation speed (how many bars between the zone activating and the entry-timeframe actually confirming) previously showed slower confirmations winning more often; on the corrected data, the slowest quartile now wins *least* often. Given it already reversed once, treat this as unstable rather than actionable in either direction.
- **The honest caveat**: the win rate still isn't fully stable across the year (now 40%–90.9% by quarter, a narrower range than before, but real variation remains, and the most recent partial quarter is the weakest) — read these patterns as "what worked in this window," not guaranteed to hold forever.

None of these filters have been coded into the actual strategy yet — this was purely an investigation to find out what's there before deciding whether to act on it. (The Pine Script dashboard displays a Friday filter and a low-volatility filter, but per the discussion above only the volatility filter still has any basis — and even that has deliberately not been wired into the live alert path yet, so it doesn't silently diverge from what's actually been backtested.)

## Where things stand / what's next

- The trend-detection and "yesterday's low" pieces are validated and could be turned into a useful tool on their own.
- The full entry/exit system shows a real edge (75 trades, 80.0% win rate, +0.44R after both the stop-loss cross-wiring fix and the order-block zone-timing look-ahead-bias fix, consistently beating random) — the case for starting Pine Script work is meaningfully stronger than the original 24-trade result, though a second instrument and a proper sensitivity check on the guessed parameters would make it even more solid before committing fully, and the sample is now smaller (75 vs the earlier, partly look-ahead-biased 153) so it deserves a bit more caution, not less.
- We now also know a concrete, testable way to potentially improve the win rate further (skip low-volatility trades, optionally also require the New York or London/NY-overlap session) — worth deciding whether to build this into the strategy. The Friday filter that used to be part of this recommendation no longer holds (see above) and should not be built in.
- **TradingView (Pine Script) work has started** — the first layer (structure detection + bias dashboard, with Friday/volatility flagged on the dashboard for information only, not yet gating alerts) is built and has been tested live on a real TradingView chart. See the dedicated section below.

## TradingView (Pine Script) tool — Phase A built and verified

**How this actually works, for clarity**: Pine Script isn't a standalone app you install — it's code that lives *inside* TradingView. You paste it into TradingView's built-in editor (the "Pine Editor") and add it to a chart, and TradingView's own servers run it against their live price feed. Alerts set up from the script keep working even when you don't have the chart open (TradingView evaluates them server-side and can notify you by app push, email, or a webhook). This is different from our Python tool, which only runs when you manually execute it on your own machine.

**We're building this in three stages, on purpose**: (1) a visual-only indicator first, so we can eyeball it against a real chart before trusting it, (2) then add the zone-detection and entry-timing logic, (3) then convert it into a strategy that can actually fire alerts. Pine Script has real, well-documented pitfalls around accidentally "seeing the future" (repainting), so verifying each layer visually before adding the next is the safer order — the same "validate before trust" approach used throughout this whole project.

**Stage 1 (done) — what it shows on the chart:**
- Swing highs/lows and Break-of-Structure / Change-of-Character labels, using the exact same close-only rule validated in Python (a wick through a level doesn't count, only a candle close does)
- A dashboard table showing the 4-hour trend, 1-hour trend, and combined bias (only set when both agree)
- The Friday flag and the current volatility regime (from the pattern analysis above), and a final "Take Trade? YES/NO" summary

**It was actually tested, not just written and assumed to work:**
- Pasted into TradingView, hit two real errors, both fixed: (1) a duplicate-declaration error from TradingView's own new-script template not being fully cleared before pasting, (2) Pine's built-in volatility-percentile function has a hard cap of 5000 historical bars, but the script asked for 5760 — trimmed the lookback window to 4800 (about 50 days instead of 60) to fit under the limit.
- Once running on a live EUR/USD 5-minute chart, the dashboard and labels behaved exactly as expected — when the 4-hour and 1-hour trends disagreed, the dashboard correctly showed "Combined Bias: none" and "Take Trade?: NO," and the BOS/CHoCH labels lined up with the real turning points on the chart.

**What's explicitly NOT built yet** — this is only the first half of the full checklist:
- Liquidity targets (session highs/lows, previous day's high/low)
- The order block + fair value gap "zone" detection
- The 5-minute lower-timeframe confirmation trigger (the actual entry signal)
- Stop-loss placement and the two-stage profit targets
- Alerts (nothing notifies you yet — this stage is look-but-don't-act)

File: `pine_scripts/smc_structure_bias_dashboard.pine`. Next steps: add the zone/confirmation layer, then convert to a full alerting strategy once that's visually verified too.

**Stage 2 — adds the rest of the checklist:**
- Liquidity targets: Asia session high/low (freeze-and-hold, ports `liquidity/sessions.py`) and the previous COMPLETED UTC calendar day's high/low (ports `liquidity/daily_weekly.py`)
- Order block + fair value gap zone detection on the 15-minute timeframe, gated on a confirmed same-direction BOS/CHoCH (ports `patterns/order_blocks.py` + `patterns/fvg.py`, wired the same way `scripts/run_backtest.py` does it) — drawn as a box anchored to the actual order-block candle
- The 5-minute confirmation trigger (chart-timeframe trend flips to match HTF bias while a same-direction zone is still active) — ports `signal/ltf_confirmation.py` + `signal/checklist.py`
- Stop-loss (order-block extreme ± an ATR buffer) and the two staged targets (session level, then previous day's level), drawn as lines once a signal fires, plus an expanded dashboard

**Stage 3 — alerts, folded into the same file:** an `alert()` call fires whenever a full checklist signal confirms, carrying entry/stop/target1/target2 in the message so it's actionable without the chart open. Wired up via TradingView's Create Alert dialog ("Any alert() function call").

File: `pine_scripts/smc_zone_entry_dashboard.pine` (supersedes Stage 1 on the chart — includes everything Stage 1 showed, plus zones/targets/confirmation, plus alerts).

**Tested on a live EURUSD 5-minute chart (2026-07-27):** two real compile bugs were caught and fixed (a duplicate-declaration/template issue in Stage 1, and `label.new` doesn't take a `location` argument — that's `yloc` — in Stage 2). Structure/bias (Stage 1) and the order-block zone detection + box drawing (Stage 2) were both visually confirmed to behave correctly against real price action. **What has NOT been observed firing live yet**: the actual entry-signal path (a full checklist fire producing the LONG/SHORT SETUP label, entry/stop/target lines, and the Stage 3 alert) — this requires 4H and 1H bias to agree, which didn't happen during the test window; a full scroll back through TradingView's entire available 5-minute history (plus a diagnostic marker for the weaker "zone matches bias" condition) didn't clearly turn one up either, most likely just because the checklist is genuinely rare (~1 signal every 5 days per the Python backtest, ~6.3 trades/month) relative to how much 5-minute history TradingView's plan makes available. The alert() wiring reuses that same untested code path — **treat the first several real signals/alerts with extra scrutiny and cross-check them against the chart** before trusting them the way Stages 1-2's visuals have already been trusted.

**A full line-by-line validation pass** (comparing the Pine script against every Python module it ports, not just visual chart checks) turned up two real stop-loss bugs — one in the Pine script, one in the "validated" Python backtest it was ported from:
- **Pine-side (fixed)**: the ATR used for the stop buffer was frozen at the order-block candle's own reading instead of using the live ATR at signal time. Fixed to match `run_backtest.py`'s `atr_5m_on_15m_scale` behavior.
- **Python-side (fixed, and it mattered)**: `run_backtest.py` had stop-loss placement cross-wired to the *opposite*-direction order block's extreme (a bearish trade's stop referenced the most recent bullish order block's high — an unrelated, possibly-stale zone — instead of the bearish zone that actually produced the trade). This Pine script never had this bug. Fixing it in Python and re-running the backtest **improved** every headline number at the time (153 trades, 75.8% win rate, +0.67R expectancy, profit factor 6.00, max drawdown -3.44R).
- **Python-side, second bug (fixed)**: a subsequent audit found that those very numbers were still look-ahead-biased — `run_backtest.py` was starting the order-block zone's "active" window at the candidate candle itself, using `ob_15m["bullish_ob"]`/`["bearish_ob"]` flags that `detect_order_blocks()` only actually knows `lookahead-1` (4) bars later (it looks forward to confirm the expansion move, FVG, and BOS/CHoCH). This Pine script's `f_zone15` never had this bug either — it was already written causally, waiting for the lookahead window to close before evaluating (`idx = lookahead - 1`; see the header comment above `f_zone15`). Fixing it in Python and re-running the backtest changed the numbers again: **75 trades, 80.0% win rate, +0.44R expectancy, profit factor 3.80, max drawdown -2.49R** — see `reports/phase1_summary.md` Section 4 for the full before/after table. The forex_trade_pattern_analysis.md winner/loser breakdown was built on the original pre-fix dataset (before both bugs were fixed) and is now stale, not yet re-verified against the corrected trades.

### A live Python monitor was considered, then set aside
Before settling on Pine Script, we discussed building a separate live-monitoring Python tool (with Telegram alerts, similar to the stock tool) that would watch forex prices in real time outside of TradingView. One small, harmless groundwork refactor was made toward this (separating the signal-generation logic in `scripts/run_backtest.py` so it could be fed live data instead of historical data), but the decision was made to prioritize the Pine Script path instead, so this wasn't built further.

## Where the actual code lives

Everything described above is in this folder (`trading_tool/`): the testing toolkit is in `smc_validator/`, the scripts that download data and run the tests are in `scripts/`, the Pine Script files are in `pine_scripts/`, and the plain-language results are written up in `reports/phase1_summary.md` (with more detail in `reports/prev_day_low_validation.md`, `reports/backtest_summary.md`, and `reports/forex_trade_pattern_analysis.md`).

## Second, separate tool: stock_hh_ll_tool (Indian stock swing-trade alerts)

A second, unrelated tool was requested alongside the forex work: something that watches a list of Indian stocks and messages you on Telegram whenever a stock confirms a new Higher High (the same "market structure" idea — HH/HL uptrend, LH/LL downtrend — applied to stocks instead of forex, for swing trading rather than intraday).

**Good news: most of the hard work was already done.** The trend/structure-detection engine built and tested for the forex project (`smc_validator/structure/`) doesn't care whether it's fed forex or stock prices, or daily vs. hourly bars — so this new tool reuses it directly rather than reinventing it.

**What's built (`stock_hh_ll_tool/`), all tested and working against live data:**
- Fetches daily (and, once a stock flags, intraday 4H/1H/30/15/5min) price data for any NSE-listed stock, free, via Yahoo Finance
- Detects overnight gaps (today's open vs. yesterday's close) — a gap-up alongside a confirmed Higher High is treated as a stronger signal
- Confirms whether *today specifically* produced a new Higher High or Lower Low (not just "is the trend up," but "did something change today"), in either direction
- **Projects the next level to watch**: once a break fires, looks back through the stock's entire history for the nearest older resistance (or support) level that hasn't been broken yet, and shows it in the alert. We checked this honestly against a random-day baseline on 10 years of real data (5 major NSE stocks) before trusting it — the hit rate isn't meaningfully better than picking any random day's nearest level, so it's shown as *useful context* ("here's the next level on the chart"), not oversold as a proven prediction. Full detail: `reports/stock_target_investigation.md`.
- For flagged stocks, checks whether the smaller timeframes agree with the bigger picture, as an entry-quality check
- Sends a Telegram message listing anything flagged; if no stock breaks out on a given day, it just says so — no false alarms
- Survives bad/delisted tickers in the watchlist without crashing the whole run

**What's still needed from you before it can run for real:**
1. Your actual watchlist of stocks (currently set to 5 placeholder names — Reliance, TCS, Infosys, HDFC Bank, ICICI Bank — in `stock_hh_ll_tool/config.yaml`)
2. A Telegram bot token + chat ID (instructions for getting these from @BotFather are in that same config file) — without these, the tool still runs and prints/saves results, it just won't message you
3. Once both of those are set, this can be scheduled to run automatically once a day after the market closes

This tool is currently paused/parked while we finish validating the main forex strategy, but it's in a working, ready-to-extend state whenever you want to pick it back up.

### Entry/exit strategy: investigated, then built
You described a specific entry/exit approach: enter on a lower-timeframe retracement-reversal after a higher-timeframe Higher High, exit on a Lower Low (trend reversal). Before building this, we ran three parallel investigations (research into established trading practice, a 35-stock/10-year backtest, and an independent skeptical stress-test) rather than assuming it would work. Full findings: `reports/stock_entry_exit_strategy_investigation.md`. Short version: it's a real, recognized strategy pattern (related to Elder's Triple Screen and Turtle Trading), it's not a bug or an overfit result, but on trending Indian large-caps it usually loses to simply buying and holding — its genuine, proven strength is **protecting capital on stocks that turn out badly** (it stayed flat instead of losing 90% on a stock that actually collapsed). You chose to build it as designed anyway, with the fixes the investigation identified.

**What's built** (`stock_hh_ll_tool/entry_exit.py`, `position_state.py`), all tested (54 tests passing project-wide) and verified on real historical data:
- **Weekly (trend) / Daily (entry timing)** timeframes — not Daily/Hourly, matching the established 4-6x timeframe-ratio guidance the research turned up (this pairing is also what both backtests validated).
- **A structure-based stop-loss** (the daily swing low at the time of entry, minus a small buffer) — the investigation found this roughly halves the worst-case single-trade loss while *slightly improving* total return, beating both "no stop" and a fixed-percentage stop.
- **Realistic order filling**: a signal detected today doesn't pretend to buy at today's already-known closing price — it fills at the *next* day's actual opening price, which is what would really happen. This needs the tool to remember, day to day, whether it's watching, has a pending entry, or is already in a position for each stock — a small state file (`stock_hh_ll_tool/state/positions.json`) now tracks this, since (unlike the simple daily HH/LL alert) this signal can span weeks.
- Position-sizing guidance (risk 1% of capital per the stop distance, shown in each entry alert) — not enforced, since the tool doesn't know your account size, just calculated and shown.
- Verified end-to-end on real INFY.NS history: a real signal fired 2024-11-22, filled at the next trading day's open (correctly skipping the weekend), and was later stopped out for a small loss — exactly the intended behavior.

This is a genuine trade-tracking signal now, not just an alert — go in expecting it to behave like the investigation found: better at avoiding disasters than at beating buy-and-hold on stocks that are already reliably compounding.
