# REGIME_EVENT_RESULTS.md - on-chain state: event studies & regimes

Two descriptive lenses (no requirement that on-chain *predict* daily returns).

> **Low-power caveat:** ~1.5 cycles in 2020-2026 -> few independent extreme events. `n_events` counts *onsets* (entering the extreme), which is small. Read as illustrative.

## Bottom line

- **Event study REFUTES 'extremes mark turns' in this sample.** Extreme-*expensive* readings preceded strong *gains*, not drawdowns (BTC MVRV expensive-extreme -> +60% at 90d / +165% at 180d; ADA MVRV -> +255% at 90d), and cheap extremes preceded flat/negative returns. Same cause as `CYCLE_TIMING_RESULTS.md`: in a 1.5-cycle uptrend, 'expensive' (new-high percentile) coincides with ongoing bull runs - momentum beats mean-reversion.
- **Regime clustering is the one (mildly) affirmative, DESCRIPTIVE result.** On-chain features partition history into states with materially different realised forward returns (BTC +5.5%..-0.9% at 30d; ADA +36.9%..-12.8%). So on-chain **describes** market phases - even though every predictive test shows it does not **forecast** direction out-of-sample. This is in-sample/descriptive (global scaling, realised returns), not a trading signal.
- **HORIZON CAVEAT (added later):** the event study uses <=180d forward returns, so it is horizon-blind to the multi-year cycle. `MVRV_SENTIMENT_RESULTS.md` shows that at ~540d the BTC valuation signal **reverses** (cheap+fear outperforms). Read the 'extremes don't mark turns' finding as short-horizon; the cycle-scale claim is illustrated (for BTC, ~1 bottom) there, not here.

## 1. Event study - do extreme valuation readings mark turns?

For each indicator, the day its look-ahead-safe percentile *enters* an extreme (>0.95 expensive / <0.05 cheap); average forward return afterwards. The claim holds if **expensive extremes -> negative** forward returns and **cheap extremes -> positive**, beyond the unconditional baseline.

Unconditional baseline mean forward return:
- **BTC**: 7d +1.0%, 30d +4.7%, 90d +16.5%, 180d +39.4%
- **ADA**: 7d +1.5%, 30d +8.1%, 90d +34.8%, 180d +93.0%

| Asset | Indicator | Event | n | fwd 7d | fwd 30d | fwd 90d | fwd 180d |
|-------|-----------|-------|---|--------|---------|---------|----------|
| BTC | MVRV | expensive extreme (>0.95) | 9 | +2.4% | +15.6% | +60.3% | +165.0% |
| BTC | MVRV | cheap extreme (<0.05) | 11 | -1.7% | -7.5% | -6.5% | +10.9% |
| BTC | Puell | expensive extreme (>0.95) | 0 | n/a | n/a | n/a | n/a |
| BTC | Puell | cheap extreme (<0.05) | 33 | +0.6% | +0.5% | -0.5% | -2.7% |
| BTC | NVT | expensive extreme (>0.95) | 22 | +2.8% | +11.1% | +52.7% | +87.1% |
| BTC | NVT | cheap extreme (<0.05) | 0 | n/a | n/a | n/a | n/a |
| ADA | MVRV | expensive extreme (>0.95) | 14 | +8.4% | +42.7% | +254.7% | +350.7% |
| ADA | MVRV | cheap extreme (<0.05) | 4 | +4.3% | +33.1% | +35.2% | +7.4% |
| ADA | NVT | expensive extreme (>0.95) | 18 | +3.8% | +25.2% | +132.6% | +299.8% |
| ADA | NVT | cheap extreme (<0.05) | 8 | -8.3% | -12.5% | -23.8% | +1.7% |

**Read:** compare each row to that asset's baseline above. An expensive-extreme row far *below* baseline (or negative) supports 'tops'; a cheap-extreme row far *above* baseline supports 'bottoms'. Small `n` = treat as anecdote, not proof.

## 2. Regime clustering (unsupervised, descriptive)

KMeans (k=4) on standardised on-chain features (MVRV, NVT, Puell, velocity, active addresses, volatility). Each regime characterised by its realised next-30d return and volatility. (Global scaling - descriptive, in-sample only.)

| Asset | Regime | days | share | mean fwd-30d | mean vol | median date |
|-------|--------|------|-------|--------------|----------|-------------|
| ADA | 2 | 260 | 11% | +36.9% | 0.068 | 2021-05-14 |
| ADA | 0 | 408 | 18% | +18.5% | 0.063 | 2020-08-22 |
| ADA | 1 | 1379 | 60% | +2.4% | 0.039 | 2024-05-02 |
| ADA | 3 | 237 | 10% | -12.8% | 0.046 | 2022-01-16 |
| BTC | 2 | 356 | 18% | +5.5% | 0.026 | 2024-09-12 |
| BTC | 1 | 855 | 44% | +3.7% | 0.030 | 2022-11-17 |
| BTC | 0 | 233 | 12% | +2.2% | 0.040 | 2021-04-26 |
| BTC | 3 | 506 | 26% | -0.9% | 0.023 | 2025-05-02 |

**Read:** if regimes separate cleanly into high-return/low-vol vs low-return/high-vol states, on-chain features *describe* market phases (a valid, if descriptive, use) - even though earlier tests show they don't *predict* direction out-of-sample.
