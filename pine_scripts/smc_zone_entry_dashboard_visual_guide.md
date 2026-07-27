# Visual Guide: SMC Zone + Entry Dashboard

What every color, line, box, and label on `smc_zone_entry_dashboard.pine` means, and how they combine to show a real trade setup. Meant to be read next to the actual TradingView chart.

## Quick reference

| What you see | Where | Color | Meaning |
|---|---|---|---|
| Faint background tint | Whole chart | Green / Red / none | 4H+1H combined bias: bullish / bearish / disagree (no tint) |
| Solid black line, labeled "Swing High" / "Swing Low" | On the 5m chart | Black | The real, tradeable swing structure (5-bar pivot) |
| Dashed black line, labeled "Internal High" / "Internal Low" | On the 5m chart | Black | Minor/noise-tier structure (2-bar pivot) — context only, not what you trade |
| Growing rectangle over a session's time window | On the 5m chart | Purple / Yellow / Blue | Asia / London / New York session range (that session's own high-low) |
| Bordered rectangle behind price | On the 5m chart | Green / Red | An active order-block zone (bullish / bearish), the "location" component of the checklist |
| Small tag reading "BOS" or "CHoCH" | Above/below candles | Lime / Blue / Orange / Purple | A structure break on the 5-minute chart itself (see breakdown below) |
| Dashed white line | Horizontal, extending right | White | Entry price of the most recent fired setup |
| Solid red line | Horizontal, extending right | Red | Stop-loss of the most recent fired setup |
| Dotted yellow line | Horizontal, extending right | Yellow | Target 1 (nearest session high/low) |
| Dotted aqua line | Horizontal, extending right | Aqua | Target 2 (previous day's high/low) |
| "LONG SETUP" / "SHORT SETUP" tag | At the entry candle | Lime / Red | A full checklist signal just fired |
| Small triangle at the bottom of the pane | Below price | Fuchsia | **Temporary diagnostic** — ignore once a real setup has fired at least once (see caveats) |
| Table, top-right corner | Fixed position | — | The dashboard — current state of every layer, in words |

## Element by element

### 1. Background tint (bias)
The whole chart background gets a very faint green or red wash depending on whether the 4-hour and 1-hour trend agree. **No tint at all means they disagree** — no trade is possible regardless of anything else on the chart, since every other layer requires this to be set first. This is the first thing to check.

### 2. Swing High / Swing Low (solid black lines)
These are the real, structural swing points — a 5-bar-left/5-bar-right pivot, the same lookback used for every BOS/CHoCH and bias calculation in the script. **This is the range the strategy actually trades between.** Each line is drawn starting at the actual candle where the pivot occurred, extends right, and is replaced by a new line the moment a fresh pivot of that type confirms — so at any time you're looking at the current "swing high" and "swing low," not a history of every past one.

### 3. Internal High / Internal Low (dashed black lines)
Same idea, but on a tighter 2-bar-left/2-bar-right lookback — this is the noise tier. It moves much more often than the swing lines because it's catching every small wiggle, not just genuine trend-defining turns. It's shown for context (to see how choppy the current leg is) — it is **not** anything the strategy trades against.

### 4. Session boxes (Asia / London / New York)
Three colored rectangles, one per session, each showing that session's own developing high-low range:
- **Asia** (light purple) — 09:00-18:00 Tokyo time
- **London** (light yellow) — 08:00-16:30 London time
- **New York** (light blue) — 08:00-17:00 New York time

Each box starts when its session starts, grows taller as new highs/lows form during the session, and freezes once the session ends. Past sessions' boxes stay on the chart (not deleted), so you can compare day to day — though TradingView will eventually start dropping the oldest ones once there's a lot of history loaded, which is normal.

### 5. Order block zone (bordered box, green or red)
This is the "location" piece of the checklist — a translucent box marking a specific 15-minute candle that qualified as an order block (a small-bodied consolidation candle right before a strong expansion move, confirmed by a matching structure break and a fair value gap). Green = bullish zone, red = bearish zone. It's anchored to the actual order-block candle (which will be a little in the past by the time it's confirmed — that's expected, not a bug) and keeps extending right for as long as it's "active" (see dashboard row 7). Once its active window expires, it stops growing and just sits there as a historical marker.

### 6. BOS / CHoCH tags (structure breaks on the 5-minute chart)
Small labeled tags marking every confirmed structure break on the chart's own (5-minute) timeframe:
- **Lime "BOS"** — bullish continuation break
- **Blue "CHoCH"** — bullish reversal break (trend was bearish, just flipped)
- **Orange "BOS"** — bearish continuation break
- **Purple "CHoCH"** — bearish reversal break (trend was bullish, just flipped)

These fire only on confirmed (closed) candles — never mid-bar — so they don't move around or disappear once drawn.

### 7. Entry / Stop / Target lines + SETUP label
These only appear once every condition in the checklist lines up on the same bar: background tint agreed (step 1), a same-colored order-block zone is still active (step 5), AND the 5-minute BOS/CHoCH structure (step 6) just flipped to match the background's direction. When that happens:
- A **"LONG SETUP"** (lime) or **"SHORT SETUP"** (red) tag appears at that candle
- Four lines extend right from that point: **white dashed = entry**, **red solid = stop**, **yellow dotted = target 1**, **aqua dotted = target 2**
- If you've set up a TradingView alert on this indicator, you also get a push/email/webhook notification with these exact numbers in the message

Only the most recent setup's lines/label are ever shown — firing a new one deletes the old lines and draws fresh ones, it doesn't stack them up.

### 8. Fuchsia diagnostic triangles (temporary)
Small triangles along the bottom of the price pane. These mark bars where an active zone's direction already matches the background bias — i.e., "close, but the exact trigger bar hasn't happened yet." This was added purely to help find a real setup while testing and **should eventually be removed from the script** once a full LONG/SHORT SETUP has actually been observed firing live. Don't read anything into it beyond "conditions are close here."

### 9. The dashboard table (top-right)
Fifteen rows, refreshed every bar, always showing the state as of the most recent bar:

| Row | What it shows |
|---|---|
| 4H Trend / 1H Trend | Each timeframe's own trend direction |
| Combined Bias | Both agree → bullish/bearish; disagree → none |
| Day Filter | Whether today is Friday — **informational only**, does not affect alerts |
| Volatility | Current 15m ATR percentile bucket (low/medium/high) — **informational only** |
| Zone (15m) | Current order-block zone's direction and whether it's still active or expired |
| Zone Bars Left | Countdown (in 15-minute bars) until the current zone's active window times out |
| Filters OK? (info only, doesn't gate alerts) | Whether bias is set AND it's not Friday AND volatility isn't low — purely a display flag, doesn't gate anything (see caveats) |
| Last Signal | Direction of the most recent fired setup, or "none yet" |
| Entry / Stop / Target 1 / Target 2 | The exact price levels of that most recent setup |
| Stage | Development status note (currently "3/3 — alerts wired (untested)") |

## How to read a live setup, start to finish

1. Watch the **background tint**. Nothing else matters until it's clearly green or red.
2. Once tinted, look for a **zone box** (green or red bordered rectangle) of the *same* color as the background, and check the dashboard's **Zone Bars Left** — it needs to still be counting down, not at 0.
3. Watch the **5-minute BOS/CHoCH tags**. You're waiting for one that matches the background's direction (lime/blue if green background, orange/purple if red background) to appear *while the zone box from step 2 is still active*.
4. When all three line up on the same candle, the **SETUP label and the four colored lines** appear, and the dashboard's **Last Signal / Entry / Stop / Target 1 / Target 2** rows update.

## Caveats — read before trusting what you see

- **The Friday and Volatility dashboard rows are informational only.** Toggling their inputs, or seeing "NO" in the "Filters OK?" row, does not stop a setup from firing or an alert from being sent. This is deliberate — the backtested, validated numbers behind this strategy don't apply either filter (see `reports/forex_trade_pattern_analysis.md` for the current recommendation on volatility filtering, and why the old Friday-filter recommendation was dropped).
- **The full entry-signal path (step 4 above) has not yet been observed firing on a live chart.** Every individual layer (structure, bias, zone detection, box drawing) has been visually confirmed against real price action, but the exact moment all three conditions align at once is rare enough that it hasn't happened yet during testing. Treat the first several real SETUP tags/alerts with extra scrutiny and check them against the chart by eye before trusting them.
- **By default, past zones and setups stay on the chart instead of being deleted** (the "Keep past zones/signals on chart (for practice)" input, on by default). Scroll back and you'll see every prior zone box, frozen where it stopped growing, plus every past SETUP label with its own entry/stop/target segment (a fixed length — `signalLineBars` bars — rather than an infinite ray, so old signals don't all smear together at the right edge of the chart). Turn that input off to go back to the original "only the single latest one" behavior. Either way, the dashboard table's Last Signal/Entry/Stop/Target rows always show only the most recent one, regardless of this setting.
