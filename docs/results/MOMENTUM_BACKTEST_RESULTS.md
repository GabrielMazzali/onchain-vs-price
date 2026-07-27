# MOMENTUM_BACKTEST_RESULTS.md - quantifying the one significant effect

The only feature that Granger-caused next-day returns anywhere in this project was
**price momentum** (`Price_vs_MA7`, BTC). This backtests the simplest rule from it:
**hold the asset when price > its N-day moving average, else cash** (long/flat, fixed
rule, acted next day - no look-ahead, no parameter fitting).

> This is a **price** strategy, not on-chain - the 'what does work' counterpoint to
> the on-chain nulls. Momentum is a well-known effect; we are only quantifying it.

Focus: **how often it wins, before fees**, then a fee sweep (0 / 10 / 25 bps). MA7 is
the Granger-significant window; 14/30/50 are robustness checks.

## Buy-and-hold reference (whole sample)

| Asset | CAGR | Sharpe | MaxDD | daily positive % | monthly positive % |
|-------|------|--------|-------|------------------|--------------------|
| BTC | +43.0% | +0.90 | -76.7% | 51.8% | 55.1% |
| ADA | +35.1% | +0.78 | -92.2% | 49.5% | 39.7% |

## Bottom line (answering: how often does it win, before fees?)

- **It wins *infrequently*.** Individual trades end positive only **18%-31%** of the time; daily hit-rate ~50% and monthly ~30-50% - no better than just holding. Momentum makes money (when it does) via **asymmetry** (a few big winners, many small losers) and by **cutting downtrends**, not by being right often.
- **The Granger-significant window (MA7) is the weakest.** Even at 0 bps it UNDERPERFORMS buy-and-hold on BTC (+28% vs +43% CAGR), and with realistic fees it collapses (+5% at 25 bps). Statistical significance did NOT translate into a tradeable edge at that frequency.
- **Longer windows (MA30/50) look better risk-adjusted** (higher Sharpe, shallower drawdown than buy-and-hold), and ADA momentum beats buy-and-hold broadly - **but** this is in-sample over ~1.5 cycles, and MA30/50 were not the Granger-tested windows (picking them is mild hindsight/multiple-testing).
- **Honest verdict:** momentum's value here is **drawdown / risk reduction**, not a high win-rate and not out-of-sample-proven excess return. It is a *price* effect - the 'what does work (a bit)' counterpoint to the on-chain nulls, not an on-chain result.

## Momentum strategy - BEFORE FEES (0 bps): how often does it win?

| Asset | MA | trade win-rate | daily hit-rate | monthly hit-rate | time in mkt | # trades | CAGR | Sharpe | MaxDD |
|-------|----|----------------|----------------|------------------|-------------|----------|------|--------|-------|
| ADA | 7 | 29.1% | 48.9% | 44.9% | 48% | 247 | +63.6% | +1.06 | -73.4% |
| ADA | 14 | 27.3% | 50.5% | 38.5% | 47% | 150 | +61.8% | +1.04 | -78.4% |
| ADA | 30 | 21.6% | 49.8% | 32.1% | 44% | 97 | +64.7% | +1.06 | -77.5% |
| ADA | 50 | 17.6% | 50.2% | 27.3% | 42% | 68 | +62.8% | +1.04 | -70.0% |
| BTC | 7 | 30.6% | 49.6% | 50.0% | 53% | 255 | +28.1% | +0.81 | -70.3% |
| BTC | 14 | 28.5% | 50.9% | 42.3% | 53% | 165 | +31.7% | +0.89 | -56.5% |
| BTC | 30 | 25.5% | 51.8% | 44.9% | 53% | 102 | +43.9% | +1.10 | -49.9% |
| BTC | 50 | 27.4% | 53.1% | 44.2% | 54% | 62 | +57.8% | +1.31 | -56.1% |

**How to read the win-rates:**
- **trade win-rate** = of all long episodes, the % that ended net positive.
- **daily hit-rate** = of days the strategy was in the market, the % that were up.
- **monthly hit-rate** = % of calendar months with a positive strategy return.

Momentum's signature is usually a **modest daily hit-rate (~50-55%) but positive
expectancy** (winners bigger than losers) and a **better drawdown** than buy-and-hold,
because it exits downtrends. Compare the win-rates and MaxDD to the buy-and-hold row.

## Fee sensitivity - when does the edge die? (MA7)

| Asset | fee (bps) | CAGR | Sharpe | MaxDD | # trades |
|-------|-----------|------|--------|-------|----------|
| ADA | 0 | +63.6% | +1.06 | -73.4% | 247 |
| ADA | 10 | +51.4% | +0.94 | -77.2% | 247 |
| ADA | 25 | +34.9% | +0.77 | -82.5% | 247 |
| BTC | 0 | +28.1% | +0.81 | -70.3% | 255 |
| BTC | 10 | +18.3% | +0.61 | -73.3% | 255 |
| BTC | 25 | +5.0% | +0.32 | -78.5% | 255 |

Each switch costs the fee; MA7 trades often, so fees bite. If CAGR/Sharpe fall below
buy-and-hold once realistic fees (10-25 bps) are applied, the statistically-real
momentum effect is **not economically tradeable** at this frequency - a common and
honest finding.
