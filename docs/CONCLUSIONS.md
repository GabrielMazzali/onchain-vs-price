# CONCLUSIONS.md — can on-chain metrics predict BTC/ADA price? (start here)

**One-line answer: No.** Over 2020-01-01 → 2026-06-01 (daily data, 2344 rows/asset),
on-chain metrics carry **no detectable incremental directional signal** for Bitcoin
or Cardano. The only thing with a real (but small, and untradeable-as-tested)
predictive footprint is **past price itself** (momentum). Adding on-chain data on
top of price does not help.

This is a **rigorous negative result**, established by three independent lines of
evidence that all point the same way. This file is the summary; the detail lives in
`VALIDATION_PLAN.md` (the plan/tracker), `METHODS_EXPLAINED.md` (plain-language guide
to every method), and `results/` (the raw evidence).

---

## The question

*Can we build a model that trades BTC or ADA using on-chain analysis?* Concretely:
do on-chain metrics (NVT, MVRV, hash rate, exchange flows, staking, active
addresses, …) predict whether the price goes up or down over the next 3–30 days,
**better than you could from price alone**?

## The answer, and how we know

Three steps, each answering a different sub-question. They **converge** — which is
what turns "my model didn't work" into "there is no signal."

### 1. Do supervised models find a signal? → No. (Phase 0)
Two walk-forward models (Logistic Regression, XGBoost) across 24 configurations
(2 assets × 4 horizons × 3 thresholds). Both barely beat guessing (MCC ≈ 0.09) and
their **cost-weighted MCC was negative** (the confident directional calls were, on
net, harmful). Crucially, the per-fold **standard deviation was as large as the mean**
— fold to fold, the "signal" was indistinguishable from noise.
→ `results/PHASE0_RESULTS.md`

### 2. Is it the on-chain data, or just the model? → The data. (Ablation)
A controlled A/B test: same model with **price-only** features (A) vs **price +
on-chain** (B), identical rows and folds, so the only difference is the on-chain
data. Across all 24 paired runs, the gain from adding on-chain was **~6× smaller than
the fold-to-fold noise**, its sign flipped randomly across models, and **nothing
survived multiple-testing correction** (1/24 crossed p<0.05 — exactly what luck
predicts — and it died under Bonferroni and FDR).
→ `results/ABLATION_RESULTS.md`

### 3. Is any signal even *in the data*, before modelling? → No. (Model-free)
Three model-free tests, each correcting a different statistical trap:
- **Lead-lag correlation** with **Newey-West** correction (for overlapping-return
  autocorrelation): ~60% of the naively "significant" correlations vanish; survivors
  are weak (|r| < 0.34) and price-shaped (valuation ratios that embed price).
- **Granger causality** (conditions on the return's own past — the fair "extra info"
  test): **no on-chain feature passes for either coin.** The only survivor is price
  momentum (`Price_vs_MA7`, BTC).
- **Mutual information** (non-linear, vs a shuffled-label null): its top hits are
  near-monotonic **clocks** (`SplyCur`, `HashRate_30d_MA`, |corr with time| ≈ 1) — a
  trend/regime confound, not prediction.
→ `results/SIGNAL_DETECTION_RESULTS.md`

### 4. What if we predict *magnitude* (regression / volatility), not direction? → Still no.
Two extra angles, because everything above was directional (classification):
- **Return magnitude (regression):** already answered by the model-free tests — Granger
  fits a *regression* of the continuous next-day return and finds nothing on-chain; the
  lead-lag correlations are on the continuous forward return. No on-chain lift.
- **Volatility (how *big* the move is, not which way):** volatility *is* modestly
  predictable (persistence tracks it), but a price-only-vs-+on-chain ablation on forward
  realized volatility shows on-chain helps **nowhere** for short horizons and **hurts the
  linear model everywhere**. The one hint — XGBoost at 30 days (BTC/ADA MSE −34%/−22%) —
  **is not significant** once the overlapping-window inflation is corrected (non-overlapping
  p ≈ 0.11–0.12), and a driver analysis shows **8 trending "clock" features reproduce
  ~80–100% of it** (`HashRate_30d_MA`, `SplyCur`) — the same regime/trend confound. So
  on-chain adds no real edge to volatility either.
→ `results/VOLATILITY_RESULTS.md`

### 5. Does on-chain work for its *intended* uses — cycle timing / describing regimes?
Practitioners use MVRV/Puell/NVT for long-horizon *cycle timing*, not daily trading —
so we tested that claim directly, plus an unsupervised regime lens:
- **Cycle timing: horizon-dependent.** At **short horizons (≤180d)** the valuation
  rule *underperformed* buy-and-hold (BTC −2.8% vs +41.5% CAGR); "expensive" preceded
  *higher* returns — momentum beat mean-reversion in a 1.5-cycle uptrend. **But at
  cycle-scale horizons (~540d) it reverses for BTC**: "cheap + fear" outperformed the
  market (+137% vs +91%) and "expensive + greed" underperformed — the textbook
  valuation effect, at the timescale these indicators are actually used on. Caveat:
  at 540d this rests on **~1 independent episode** (the 2022 bottom) and **does not
  replicate on ADA** — illustrative, not statistically powered. Testing it properly
  needs multiple cycle bottoms (history back to ~2011). → `results/CYCLE_TIMING_RESULTS.md`, `results/REGIME_EVENT_RESULTS.md`, `results/MVRV_SENTIMENT_RESULTS.md`
- **The one affirmative (descriptive) result:** on-chain features *do* partition
  history into **regimes** with materially different realized returns (BTC +5.5%…−0.9%
  at 30d; ADA +36.9%…−12.8%). So on-chain **describes** market phases — even though it
  does not **predict** direction out-of-sample. (In-sample/descriptive, not a signal.)

### 6. The one effect that *is* real — is it tradeable? (price momentum)
The only feature to pass Granger was **price momentum** (`Price_vs_MA7`, BTC). Backtested
as a hold-above-MA rule: it **wins infrequently** (per-trade win-rate ~18–31%, daily/monthly
hit-rate ~50% — no better than holding); it profits via asymmetry (few big winners, cut
downtrends). The Granger-significant MA7 window **underperforms buy-and-hold even before
fees** and dies with them; longer windows (MA30/50) improve *risk-adjusted* return
(BTC MA50 Sharpe 1.31 vs 0.90) but that's in-sample/hindsight. **Takeaway:** momentum's
value is drawdown reduction, not a high win-rate — and it's a *price* effect, not on-chain.
→ `results/MOMENTUM_BACKTEST_RESULTS.md`

### 7. Does deep learning change it? → No — and it exposes a famous illusion (LSTM)
A final capacity check: an **LSTM** (sequence model) and a **DLinear** linear baseline on the
same walk-forward harness (BTC 30d/1%). Two findings:
- **No edge, and the ceiling is the data.** LSTM per-fold F1 = 0.507 ± **0.342** (Buy) /
  0.210 ± **0.330** (Sell) — the std ≈ the mean again (unstable, Phase-0 signature). Its MCC
  (~0.17) doesn't cleanly beat XGBoost (balanced F1 0.36 vs 0.41), it just leans on the
  majority Buy class (34% opposite-direction errors). More capacity climbs from *terrible*
  (DLinear MCC 0.06) to *weak-and-unstable* (LSTM/XGB ≈ 0.1–0.2) — never to *good*. So a
  deeper model can't manufacture signal that isn't there.
- **The persistence illusion, demonstrated.** Training the LSTM to predict the **price level**
  gives **R² = 0.98** — the exact impressive number crypto-DL papers report (our Transformer+GNN
  reference: R² = 0.994). But predicting **returns** it scores **R² = −0.74**, **52%** directional
  (coin flip), and **negative skill vs a naive "tomorrow = today"** baseline. The gorgeous price
  chart is just the input shifted one day. → `pipeline/4_LSTM_model_pipeline.ipynb`, `LSTM_EXPLAINED.md`

### Methodological contribution
Every rigorous null here had a **naive version that looked positive** — and each was
chased down. `CONFOUNDS.md` catalogues the six traps (overlap-inflated p, multiple
testing, the trend/clock confound, missing controls, leakage) with the concrete
false-positive each produced *in this project* and its fix. That "how to correctly
evaluate a crypto predictive claim" checklist is a contribution in its own right — the
negative result is trustworthy precisely *because* every naive positive was explained.

---

## What this means (and doesn't)

- **It means:** for BTC/ADA at daily frequency, on-chain metrics do not give a
  directional trading edge over price. Every surviving effect traces back to price
  (valuation mean-reversion, momentum) or the calendar.
- **It does *not* mean the models were a mistake.** The models are the evidence: they
  show the obvious supervised approach fails, which is what motivates — and is then
  explained by — the ablation and model-free tests. A study that skipped straight to
  Granger would be weaker.
- **It does *not* mean price-only is good.** Price-only is merely the *ceiling*, and
  the ceiling is low (F1 ≈ 0.42, MCC ≈ 0.07). The one genuinely real effect (BTC
  momentum, Granger p ≈ 1e-9) is statistically detectable but small — and **we have
  not shown it is profitable after fees.**

## Honest caveats / what is not done

- These are **statistical** results. No economic backtest with transaction costs yet
  (deferred — a flat signal isn't worth backtesting).
- The mild, non-significant lean for **XGBoost on ADA** rides on staking/supply
  features. Before *any* ADA-side claim, the Blockfrost epoch→daily expansion must be
  audited for look-ahead leakage (`VALIDATION_PLAN.md` Phase 3.3).
- Labels are fixed-horizon (e.g. 1% / 30d), which makes "Hold" rare (~6%) — the task
  is effectively Buy-vs-Sell. A **triple-barrier** relabel is the next robustness
  check to rule out a label artifact (`VALIDATION_PLAN.md` step "label redo").

## What's next (optional, for completeness)

1. **Label redo** (triple-barrier / sign-of-return) — rule out that the fixed label
   hid a signal. Expectation: low, but closes the loop for the TCC.
2. **ADA staking leak audit** (Phase 3.3) — before trusting any ADA on-chain feature.
3. **Momentum backtest** (optional) — turn the one real effect (BTC momentum) into a
   "tradeable or not, after fees?" answer.

---

## Where to read more

| I want… | Read |
|---------|------|
| The plan, phase status, and what's next | `VALIDATION_PLAN.md` |
| What each statistical method means + the **feature data dictionary** | `METHODS_EXPLAINED.md` |
| Phase 0 trustworthy-baseline numbers | `results/PHASE0_RESULTS.md` |
| The ablation tables (price-only vs +on-chain) | `results/ABLATION_RESULTS.md` |
| The model-free tests (lead-lag / Granger / MI) | `results/SIGNAL_DETECTION_RESULTS.md` |
| The volatility / regression-framing test | `results/VOLATILITY_RESULTS.md` |
| Cycle-timing test (MVRV/Puell/NVT) | `results/CYCLE_TIMING_RESULTS.md` |
| Regime clustering + event studies | `results/REGIME_EVENT_RESULTS.md` |
| Momentum backtest (the one significant effect) | `results/MOMENTUM_BACKTEST_RESULTS.md` |
| MVRV × sentiment "buy the fear?" (horizon-dependent) | `results/MVRV_SENTIMENT_RESULTS.md` |
| Deep learning (LSTM + DLinear) + the persistence illusion | `../pipeline/4_LSTM_model_pipeline.ipynb`, `LSTM_EXPLAINED.md` |
| **The confounds / methodology chapter** (how naive analysis lies) | `CONFOUNDS.md` |
| To reproduce | `../README.md` and `../CLAUDE.md` (commands); `utils/{ablation,signal_detection,volatility_test,cycle_timing,regime_events,momentum_backtest}.py` |
