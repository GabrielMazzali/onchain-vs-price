# CONFOUNDS.md - how naive analysis would have "found" an on-chain edge (and why it's wrong)

This is the methodological heart of the project. Every rigorous negative result here
had a **naive version that looked positive**. A careless analysis of the same data
would have concluded "on-chain predicts crypto" - and been wrong each time.

This file catalogues the six traps we hit, with the concrete false-positive each one
produced *in this project*, the correction, and the lesson. It doubles as a checklist
for evaluating any predictive claim on financial time series.

> For each: **the trap -> what it faked here (numbers) -> the fix -> the lesson.**
> Companion results: `results/ABLATION_RESULTS.md`, `results/SIGNAL_DETECTION_RESULTS.md`,
> `results/VOLATILITY_RESULTS.md`, `results/CYCLE_TIMING_RESULTS.md`, `results/REGIME_EVENT_RESULTS.md`.

---

## Trap 1 - Overlapping-window autocorrelation inflates significance

**The trap.** A forward return (or forward volatility) over *h* days, computed each
day, means adjacent observations share *h*-1 of their *h* days. They are almost
copies, not independent. Standard p-values assume independence, so they behave as if
there are ~*n* independent data points when there are really only ~*n/h*. Result:
absurdly tiny p-values for effects that are actually noise.

**What it faked here.**
- **Lead-lag correlations:** naive iid p-values flagged **67 (BTC) / 77 (ADA)**
  "significant" feature->return correlations. Newey-West HAC correction left only
  28 / 23, all weak and price-shaped.
- **Volatility, 30d:** the on-chain "improvement" had naive p = **2e-18 (BTC), 3e-6
  (ADA)**. The valid non-overlapping p was **0.11 / 0.12 - not significant.**

**The fix.** Newey-West (HAC) standard errors, or a non-overlapping / block subsample
(only every *h*-th point). Both collapse the fake significance.

**Lesson.** *Never trust an iid p-value on overlapping forward returns.* If you predict
an *h*-day-ahead quantity daily, your effective sample is ~*n/h*.

---

## Trap 2 - Multiple testing (trying many things, cheering the best)

**The trap.** Run enough configurations and, under pure noise, ~5% cross p<0.05 by
luck. Pointing at the best one is not a discovery.

**What it faked here.**
- **Ablation:** across 24 configs, exactly **1** crossed p<0.05 (LR ADA 14d/0.5%,
  p=0.017) - precisely the ~1.2 false positives expected under the null. It **died
  under Bonferroni (alpha=0.002) and Benjamini-Hochberg FDR (0 survive).**
- **Best-of-24 cherry-pick:** the single best classifier config (ADA 7d/0.5%, Sell
  F1 0.62) looks great until you remember it is the max of 24 tries.

**The fix.** Pre-register one primary config, or correct for the number of tests
(Bonferroni / FDR / Deflated Sharpe). Report the *full* spread, not the maximum.

**Lesson.** The headline number must survive a multiple-testing correction, or be
reported as "best of N" with N stated.

---

## Trap 3 - The trend / clock confound (features that secretly encode time)

**The trap.** A feature that rises almost monotonically over time (supply, hash-rate
MA, an expanding percentile in a bull market) is effectively a *timestamp*. Because
outcomes cluster by era (2021 up, 2022 down), any method that measures raw
*dependence* will "find" a relationship that is really just "the feature knows what
year it is."

**What it faked here.**
- **Mutual information:** the top-MI features were near-perfect clocks - `SplyCur`
  (|corr with time| = 1.00), `HashRate_30d_MA` (0.99), activity metrics (0.85). Their
  MI with the return sign is a regime artifact, not prediction.
- **Volatility, 30d:** the on-chain "gain" was reproduced ~80-100% by just the 8
  trending features (led by `HashRate_30d_MA`, `SplyCur`); non-trending on-chain
  features did far less.
- **Cycle-timing / event study:** an *expanding percentile* on a trending price
  series conflates "expensive" with "making new highs / in an uptrend" - which is why
  "expensive extremes" preceded *gains*, not drawdowns, in this sample.

**The fix.** Measure the feature's correlation with time; condition on the past
(Granger); or de-trend / difference. If removing the trending features kills the
effect, the effect was the trend.

**Lesson.** In a trending market, "predictive" and "monotone-in-time" are easily
confused. Always check `|corr(feature, time)|`.

---

## Trap 4 - No baseline / no control (crediting price effects to on-chain)

**The trap.** A model with on-chain features "works" a bit - but so would a model
with only price. Without a price-only control you cannot attribute the performance to
on-chain. Worse, several "on-chain" metrics **embed price** (MVRV = market/realized
value; NVT = market cap / tx; issuance-in-USD) - so their apparent signal is a price
signal in disguise.

**What it faked here.**
- **Phase 0 models** looked like they "did something" (F1 ~0.4) - but the **ablation**
  showed price-only matched them; on-chain added nothing (delta ~6x smaller than
  noise).
- The lead-lag survivors after Trap-1 correction were **NVT / MVRV / IssTotUSD** -
  price-embedding valuation ratios, not independent on-chain information.

**The fix.** Always run a price-only (and ideally a technical-analysis-only) control,
identical harness, and report the *difference*. Know which "on-chain" features contain
price.

**Lesson.** The question is never "does the model work?" but "does it beat the price-
only baseline?"

---

## Trap 5 - Leakage: global scaling, look-ahead, and overlapping-label CV

**The trap.** Fitting a scaler (or any statistic) on the whole dataset before
splitting leaks test-window information into training. Using future data to build a
feature is look-ahead. Overlapping labels leak between train and test folds.

**What it faked / how we avoided it here.**
- **Per-fold scaling only** (`fit_scaler_on_train`), never a global `fit_transform` -
  a documented invariant of `src/feature_engineering.py`.
- **No synthetic labels:** `classify_signal` returns NaN for NaN so the last *h* rows
  are dropped, never mapped to "Hold."
- **Walk-forward with `TimeSeriesSplit`** inner CV - never shuffled K-fold on a time
  series.
- **Look-ahead-safe features** in the cycle/event tests (expanding percentile uses
  only past+present).
- **Still open:** ADA staking (Blockfrost per-epoch spread to daily) must be audited
  for future-epoch leakage before ADA on-chain results are trusted
  (`VALIDATION_PLAN.md` Phase 3.3).

**The fix.** Fit every transform inside the training fold; forward-fill only from the
past; use purged/embargoed CV when labels overlap.

**Lesson.** Leakage almost always makes results look *better*. If a result seems too
good, suspect leakage first.

---

## Trap 6 - The persistence illusion (predicting *levels* instead of *returns*)

**The trap.** Train any model to predict the **price level** and it learns the trivial rule
*"tomorrow ≈ today."* The prediction is just the input shifted one day, so the overlay chart
is gorgeous and R² ≈ 0.99 — but it has **zero** predictive value and never beats a naive
"tomorrow = today" baseline. This is *the* most common error in crypto deep-learning papers.

**What it faked here — demonstrated with our own LSTM** (`pipeline/4_LSTM_model_pipeline.ipynb`):

| Evaluated on… | Result | Meaning |
|---------------|--------|---------|
| **Price level** | **R² = 0.98** | the illusion — "looks 98% accurate" |
| **Returns** | **R² = −0.74** | the truth — worse than predicting the mean |
| Directional accuracy | **0.52** | a coin flip |
| Skill vs naive ("tomorrow = today") | **−0.74** | negative — it *loses* to the naive baseline |

Our Transformer+GNN reference paper reports **R² = 0.9941** for Bitcoin — almost certainly this
illusion, measured on levels with no naive baseline.

**The fix.** Predict **returns**, not levels. Always report skill **relative to a naive
baseline** (persistence / random walk), plus directional accuracy. If the model can't beat
"tomorrow = today," it has learned nothing — no matter how good the price chart looks.

**Lesson.** A beautiful predicted-vs-actual **price** curve is not evidence. The only honest
question is: *does it beat the naive baseline on returns?* Here, decisively, no.

---

## Why this is a contribution, not just caveats

Put together, Traps 1-6 each independently manufactured a "positive" result from data
that, analysed correctly, shows **no edge** — including a deep-learning model that
"predicts Bitcoin with R² = 0.98" (Trap 6). A reader can therefore treat this project as a
**worked example of how to (and how not to) evaluate a predictive claim in crypto**: run
controls (Trap 4), correct for overlap (Trap 1) and multiple testing (Trap 2), check the
trend confound (Trap 3), prevent leakage (Trap 5), and predict returns-not-levels against a
naive baseline (Trap 6). The negative result is only trustworthy *because* every naive
positive was chased down and explained.

**Checklist (portable to any such study):**
1. Price-only (and TA-only) control - is the effect incremental?
2. Overlap-aware significance (Newey-West / non-overlapping) on any forward-horizon test.
3. Multiple-testing correction; report the full spread, not the max.
4. `|corr(feature, time)|` on every "important" feature; de-trend if high.
5. Per-fold scaling, no synthetic labels, walk-forward CV, no look-ahead features.
6. Predict **returns, not price levels**; report skill vs a **naive baseline** + directional accuracy.
