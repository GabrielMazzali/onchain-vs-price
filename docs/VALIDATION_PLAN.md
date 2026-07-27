# Validation Plan — can on-chain metrics predict BTC/ADA price?

> Phased research plan and process record for the TCC (XP Educação, Bacharelado em
> Ciência de Dados). For the statistical methods referenced here, see
> `METHODS_EXPLAINED.md`; for the final synthesis, see `CONCLUSIONS.md`.

## The question

Can on-chain metrics give a directional trading edge for BTC or ADA over the next
3–30 days, **beyond what price alone already tells you**?

## The core decision (2026-06-26)

Two supervised models — Logistic Regression (`2a`) and XGBoost (`2b`) — both scored
near chance (MCC ≈ 0.09), with per-fold standard deviation as large as the mean and a
**negative** cost-weighted MCC (the confident directional calls were, on net, harmful).

The key realization: a failing model cannot distinguish *"there is no signal in the
data"* from *"the model is too weak."* So the question splits in two:

- **(A) Is there *any* on-chain → future-price signal in the data?** — a model-free
  question, and the real science.
- **(B) Did our supervised setup capture it?** — answered: **no** (Phase 0).

Rather than test more models (Random Forest, LightGBM — which also cannot separate the
two cases), we pursued (A) directly with a controlled ablation and model-free tests.

---

## Phases and status

### Phase 0 — Trustworthy baseline ✅ done (2026-06-22)
Re-ran `2a`/`2b` end to end, leak-free (per-fold scaling, no synthetic labels,
`TimeSeriesSplit` inner CV), seeds pinned. Reported mean ± std across folds. **Key
finding:** per-fold std ≈ mean and MCC is undefined in ~40% of folds — fold to fold the
signal is indistinguishable from noise. Data range 2020-01-01 → 2026-06-01, 2344
rows/asset. → `results/PHASE0_RESULTS.md`

### Phase 1 — Significance + economic backtest (deferred, confirm-only)
Deferred deliberately: there is no value in permutation-testing a flat, unstable MCC or
backtesting a non-signal. To be run as a short confirmation only if a later step finds a
pulse.
- **1.1 Significance test** — permutation test (shuffle labels ~1000×) plus a
  Pesaran–Timmermann directional test → p-values for Buy/Sell, both assets.
- **1.2 Economic backtest** — signal → positions → equity curve; Sharpe, Sortino, max
  drawdown, CAGR, hit-rate vs buy-and-hold; fee sweep 0 / 10 / 25 bps.

### Phase 2 — Prove on-chain is the reason
- **2.1 Ablation** ✅ done (2026-06-30) — price-only (A) vs price + on-chain (B), same
  rows and folds, paired delta B − A plus Wilcoxon. On-chain adds **no stable,
  significant edge**: the effect is ~6× smaller than the fold-to-fold noise, 0/24 configs
  beat their own std, nothing survives Bonferroni/FDR, and the sign is incoherent across
  models. → `results/ABLATION_RESULTS.md`
- **2.2 Beat classic technical analysis** — future work: compare on-chain against a
  TA-only feature set (RSI, MACD, MA-cross) to test whether it beats cheap price-only TA.
- **2.3 Multiple-testing correction** ✅ applied — across 24 configs the best result
  (ADA 7d/0.5%) is partly luck; corrected with Bonferroni and BH-FDR (nothing survives),
  and the full 24-config spread is reported rather than the single best.

### Model-free signal detection ✅ done (2026-06-30)
Lead-lag correlation (Newey-West, overlap-aware), a hand-rolled Granger F-test, and
mutual information vs a shuffled-label null — all FDR-corrected. Converges with the
ablation: after correcting for overlap, conditioning on the return's own past (Granger
clears **every** on-chain feature; only `Price_vs_MA7` survives), and recognizing the
trend/clock confound in MI, **no incremental on-chain signal remains**. Every surviving
effect is price-driven or a calendar artifact. → `results/SIGNAL_DETECTION_RESULTS.md`

### Additional angles ✅ done
- **Volatility / regression framing** — on-chain gives no significant edge on forward
  realized volatility either; the one 30d hint is non-significant after overlap
  correction (p ≈ 0.11–0.12) and ~80–100% reproduced by trending clock features.
  → `results/VOLATILITY_RESULTS.md`
- **Cycle timing** — MVRV/Puell/NVT valuation-percentile timing underperforms
  buy-and-hold in-sample at short horizons; it reverses toward the textbook valuation
  effect only at ~540d for BTC (one independent episode, does not replicate on ADA).
  → `results/CYCLE_TIMING_RESULTS.md`, `results/MVRV_SENTIMENT_RESULTS.md`
- **Regime detection + event studies** — the event study refutes "extremes mark turns,"
  but regime clustering is **descriptively affirmative**: on-chain features partition
  history into states with materially different realized returns.
  → `results/REGIME_EVENT_RESULTS.md`
- **Momentum backtest** — the one Granger-significant effect (BTC `Price_vs_MA7`) wins
  infrequently, underperforms buy-and-hold at the significant window, and dies with fees.
  → `results/MOMENTUM_BACKTEST_RESULTS.md`

### Deep learning (LSTM + DLinear) ✅ done (2026-07-12)
A capacity-ceiling check on the same walk-forward harness. The LSTM's per-fold F1 std ≈
mean (unstable); its MCC (~0.17) does not cleanly beat XGBoost; the DLinear linear
baseline is the floor — so capacity is not the missing ingredient. Persistence-illusion
demo: R² = 0.98 on price **levels** vs −0.74 on **returns**, with negative skill against a
naive "tomorrow = today" baseline. → `../pipeline/4_LSTM_model_pipeline.ipynb`,
`LSTM_EXPLAINED.md`

### Confounds / methodology chapter ✅ done
`CONFOUNDS.md` catalogues the **six traps** (overlap-inflated p, multiple testing, the
trend/clock confound, missing control, leakage, and the deep-learning persistence
illusion) with the concrete false-positive each produced in this project and its fix.
This is the core methodological contribution.

### Phase 3 — Robustness (mostly future work)
- **3.1 Regime split** — evaluate bull vs bear separately; track rolling MCC over time.
- **3.2 Calibration + abstain** — calibrate probabilities and trade only when confident;
  plot edge vs coverage (addresses the "confident and wrong" failure).
- **3.3 ADA staking leak audit** — the one material leakage risk: Blockfrost staking is
  per-epoch, expanded to daily. If a future epoch value bleeds into earlier days, ADA's
  staking/supply features would be inflated. **Must be audited before any ADA-side
  on-chain claim** — ADA's top features are all staking/supply. Not yet done.
- **3.4 Overlapping-label CV** — 30d forward-return labels on neighboring rows share
  future information; a purged + embargo CV would remove train rows whose label window
  touches the test window. (The model-free tests already correct for overlap in their
  significance estimates.)

### Phase 4 — Further extensions (future work)
More models (Random Forest, LightGBM, an ARIMA/AR forecast floor); smarter labels
(triple-barrier, meta-labeling, regression targets); better validation (Combinatorial
Purged CV, Diebold–Mariano, White Reality Check / Hansen SPA); deeper causality/feature
study (transfer entropy, SHAP-over-time drift); and broader scope (more assets,
cross-sectional ranking, position sizing, regime-switching models, higher frequency).

---

## Overall status

Direction, magnitude, volatility, cycle-timing, and deep learning were all tested:
on-chain shows **no significant predictive edge anywhere**. The only affirmative use is
**descriptive** regime segmentation. Final synthesis in `CONCLUSIONS.md`; methodological
contribution in `CONFOUNDS.md`.

## Not done / open for future work

- **Label redo** (triple-barrier / sign-of-return) to rule out a fixed-horizon label
  artifact — the 1%/30d label makes Hold only ~6%, so the task is effectively Buy-vs-Sell.
- **ADA staking leak audit** (Phase 3.3) before trusting any ADA on-chain feature.
- **Economic backtest with transaction costs** (Phase 1.2) — deferred while the signal is
  flat.
