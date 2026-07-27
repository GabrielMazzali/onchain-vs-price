# METHODS_EXPLAINED.md — the statistics, in plain language

A friendly, no-jargon guide to every statistical method used in this project
(Phase 0, the ablation, and the model-free signal detection). Written to be read
before a research defense: if a professor asks *"what is a Newey-West correction and
why did you use it?"*, the answer is here.

Each method follows the same little structure: **what it is · why we used it ·
what we found.** Companion result files: `results/PHASE0_RESULTS.md`,
`results/ABLATION_RESULTS.md`, `results/SIGNAL_DETECTION_RESULTS.md`.

---

## The data — which features we used (data dictionary)

All features come from **daily** data, 2020-01-01 → 2026-06-01 (2344 rows/asset).
Sources: **Coin Metrics** community API (no key) for market/on-chain metrics, and
**Blockfrost** for ADA staking. Features are either **raw** (a column straight from
the source) or **engineered** (derived in `src/feature_engineering.py`). The
canonical definitions live in that file; this table is the human-readable mirror.

For the ablation, features split into two groups:
**price-only** (Model A) and **on-chain** (everything else, Model B minus A).

### Price-only features (Model A) — 6, derivable from price alone

| Feature | Raw/Eng. | Meaning |
|---------|----------|---------|
| `Log_Return` | eng. | daily log price return |
| `Price_vs_MA7` | eng. | price ÷ its 7-day moving average (short-term momentum) |
| `Price_Dist_lag_30d` / `_90d` / `_182d` | eng. | price ÷ price N days ago (medium/long momentum) |
| `Volatility_30d` | eng. | 30-day rolling std of log returns |

### On-chain features (part of Model B) — BTC 37, ADA 30

*Note: **20 of each** are just lagged copies (`_lag_{1,3,7,14,30}d`) of four base
series — `AdrActCnt`, `NVT_Tx_Basis`, `Tx_Intensity`, `Velocity_Momentum` — so the
distinct information is smaller than the raw count suggests.*

| Category | Feature | Raw/Eng. | Source | BTC | ADA | Meaning |
|----------|---------|----------|--------|-----|-----|---------|
| Network activity | `AdrActCnt` | raw | Coin Metrics | ✓ | ✓ | active addresses per day |
| Network activity | `Tx_Intensity` | eng. | Coin Metrics | ✓ | ✓ | transactions ÷ active addresses |
| Network activity | `Activity_Velocity` | raw | Coin Metrics | ✓ | ✓ | money velocity (turnover) |
| Network activity | `Velocity_Momentum` | eng. | Coin Metrics | ✓ | ✓ | 7-day % change of velocity |
| Valuation | `CapMVRVCur` | raw | Coin Metrics | ✓ | ✓ | MVRV = market value ÷ realized value |
| Valuation | `NVT_Tx_Basis` | raw | Coin Metrics | ✓ | ✓ | Network-Value-to-Transactions ratio |
| Miners / supply | `HashRate` | raw | Coin Metrics | ✓ | — | mining hash rate |
| Miners / supply | `HashRate_30d_MA`, `HashRate_vs_MA30` | eng. | Coin Metrics | ✓ | — | hash-rate 30d trend and deviation |
| Miners / supply | `IssTotUSD` | raw | Coin Metrics | ✓ | — | total daily issuance (USD) |
| Miners / supply | `Puell_Multiple` | eng. | Coin Metrics | ✓ | — | issuance ÷ its 365-day MA (miner-revenue cycle) |
| Fees | `FeeTotNtv` | raw | Coin Metrics | ✓ | — | total fees (native units) |
| Exchange flows | `FlowInExUSD`, `FlowOutExUSD` | raw | Coin Metrics | ✓ | — | coins moving to / from exchanges (USD) |
| Exchange flows | `ExchangeNetFlow`, `ExchangeNetFlow_7d_MA` | eng. | Coin Metrics | ✓ | — | net exchange flow (in − out) and its 7-day MA |
| Supply / staking ⚠️ | `SplyCur` | raw | Coin Metrics | — | ✓ | circulating supply |
| Supply / staking ⚠️ | `ActiveStakeADA` | raw | **Blockfrost** | — | ✓ | active staked ADA |
| Supply / staking ⚠️ | `StakingRatio` | eng. | Blockfrost | — | ✓ | staked ÷ circulating supply |
| Sentiment | `FearGreedValue` | raw | Coin Metrics | ✓ | ✓ | Fear & Greed index (see caveat) |
| Lagged copies | `AdrActCnt`, `NVT_Tx_Basis`, `Tx_Intensity`, `Velocity_Momentum` × {1,3,7,14,30}d | eng. | — | ✓ | ✓ | past values of the four base series |

**Two caveats to state in the write-up:**
- `FearGreedValue` is *market sentiment*, not strictly on-chain, yet it sits in the
  on-chain/non-price bucket (Model B). Harmless, but worth disclosing.
- ADA's `SplyCur` / `ActiveStakeADA` / `StakingRatio` are the features that (a) need
  the Blockfrost epoch→daily **leakage audit** before being trusted and (b) appeared
  as the **trend/clock confound** in the mutual-information test — they rise almost
  monotonically over time.

*(Building-block raw columns `CapMrktCurUSD`, `TxCnt`, and the `asset` tag are dropped
by `engineer_features` after the ratios above are computed, to avoid feeding both a
ratio and its ingredients.)*

---

## Part 0 — Two building blocks everything rests on

### Correlation
- **What:** a number from −1 to +1 for *"when one thing goes up, does the other
  tend to go up (+) or down (−)?"* `0` means no relationship.
- Two flavors we used: **Pearson** (catches *straight-line* relationships) and
  **Spearman** (catches any *same-direction* relationship even if curved, and is
  less fooled by outliers because it works on ranks).
- **Example:** correlation `−0.15` between a metric and next month's return =
  "when the metric is high, returns are *slightly* lower on average — but weakly."

### p-value
- **What:** answers *"could this result just be luck?"* Precisely: **if nothing
  real were going on, how often would I see a result this strong by pure chance?**
- Small p (e.g. < 0.05) = "luck would rarely produce this → probably real."
  Large p = "luck explains this easily → probably nothing."
- **Why it's central:** almost every test below is just a careful way of computing
  a p-value for *"is there really a signal, or am I fooling myself?"*
- **Coin analogy:** 7 heads in 10 flips — rigged coin, or normal luck? The
  p-value is exactly the chance of getting 7+ heads from a fair coin.

---

## Part 1 — How we *scored* the models (Phase 0)

The models output a daily label: **Buy / Sell / Hold**. These grade the guesses.

### F1 score
- **What:** a 0–1 grade for how well the model predicts one class (e.g. "Buy").
  Balances two mistakes — false alarms (crying Buy when wrong) and misses (missing
  real Buys). Near 1 = great, near 0 = terrible.
- **Found:** ~0.4 — mediocre.

### MCC (Matthews Correlation Coefficient)
- **What:** a stricter, more honest grade from −1 to +1. Unlike F1, it can't be
  faked by exploiting a common class; it only looks good if the model is genuinely
  right across all classes. `0` = no better than guessing.
- **Found:** ~0.09 — barely above pure guessing.

### Weighted MCC (our custom version)
- **What:** MCC that **punishes the expensive mistake 2×**. In trading, predicting
  "Buy" when the truth is "Sell" is a disaster (wrong in both directions), far
  worse than predicting "Hold" and sitting out. So opposite-direction errors count
  double.
- **Found:** **negative** — the confident directional calls were, on net, harmful.

### Walk-forward validation
- **What:** the honest way to test a time-based model — never let it see the
  future. Train on days 1–600, test 601–630; then train 1–630, test 631–660; keep
  marching forward. Each test chunk is a **"fold."**
- **Why:** mirrors real life — you only ever have the past to predict tomorrow.

### Mean ± standard deviation across folds ("never one number")
- **What:** report not just the average score across folds but also the **standard
  deviation (std)** — how much it jumps around fold to fold.
- **Why:** a good *average* can hide a model that's brilliant in some folds and
  awful in others — i.e. unstable and useless.
- **Found (the key Phase 0 result):** the std was ≈ as big as the mean
  (e.g. F1 = 0.47 ± 0.30). Fold-to-fold the "signal" was basically noise wearing a
  nice average.

---

## Part 2 — Testing "is it real, or luck?"

### Permutation / shuffle test
- **What:** break the link on purpose. Randomly **shuffle the labels** (destroying
  any real relationship), score the model, repeat ~300× → a "pile" of scores that
  represent *pure luck*. If your real score sits inside that pile, it was luck; if
  it stands far above the whole pile, it's probably real.
- **Used for:** mutual information (the shuffles are the "null" it must beat).

### Wilcoxon signed-rank test
- **What:** a paired, fold-by-fold check — *"are the differences between model B and
  model A consistently on one side of zero, or do they just wobble around zero?"*
  Doesn't assume the differences follow a bell curve.
- **Found (ablation):** p ≈ 0.8 — no reliable difference between price-only and
  price+on-chain.

### Multiple-testing correction
- **The trap:** run **24 experiments** and even with *no real signal*, about 1 will
  randomly cross "p < 0.05." Cheering that one is a mistake — it's expected luck,
  not a discovery. (The "if enough people play the lottery, someone wins" problem.)
- **Two fixes:**
  - **Bonferroni** — strict: divide the threshold by the number of tests
    (0.05 ÷ 24 ≈ 0.002). Only results beating *that* count.
  - **FDR / Benjamini–Hochberg** — smarter and less brutal: controls the *fraction*
    of your "discoveries" allowed to be false alarms.
- **Found (ablation):** exactly **1 of 24** crossed p<0.05 (= what luck predicts),
  and it **vanished** under both corrections. A mirage.

---

## Part 3 — The ablation: a controlled experiment

- **What:** an A/B test borrowed from medicine. Like a drug trial — treatment vs
  placebo, *everything else identical*, so any difference must be due to the drug.
  - **Model A (placebo):** price-based features only.
  - **Model B (drug):** price **+ on-chain** features.
  - Same days, same folds, same everything. The **only** difference is on-chain
    data, so `score(B) − score(A)` measures *exactly* what on-chain contributes.
- **Why:** it directly separates *"on-chain is useless"* from *"our model is dumb"* —
  the whole research question in one experiment.
- **Found:** the difference was tiny, ~6× smaller than the fold-to-fold wobble, its
  sign flipped randomly across models, and nothing survived multiple-testing.
  **The "drug" does nothing.**

---

## Part 4 — Model-free tests: is the signal even *in the data*?

The ablation showed *our models* can't use on-chain data. These three tests skip
models entirely and interrogate the raw data — maybe a signal is there and the
models are just too weak to see it? **Each test has a blind spot we had to correct
for — that's the interesting part.**

### Test 1 — Lead-lag correlation (+ the overlap trap + Newey-West fix)
- **What:** correlate a metric *today* with the return over the *next* 7/14/30 days.
  Does today's metric lead future price?
- **The trap — overlapping windows:** the naive p-value screamed "significant!"
  (p as tiny as 1e-62). But "next 30 days" starting today and starting tomorrow
  share 29 of the same days — so consecutive data points are almost copies, not
  independent. The p-value formula *assumes independence*, so it thinks it has far
  more evidence than it does. *(Like interviewing 100 people but 99 are the same
  person — you don't really have 100 opinions.)*
- **The fix — Newey-West:** a corrected standard error that accounts for this
  overlap/autocorrelation and deflates the fake confidence.
- **Found:** ~60% of the "significant" findings evaporated after the fix; what
  remained was weak and price-based anyway.
- **Key lesson:** *never trust a naive p-value on overlapping forward returns.*

### Test 2 — Granger causality (the strictest, cleanest test)
- **What:** *does the past of the on-chain metric help predict tomorrow's return
  **beyond** what the return's own past already predicts?* The phrase "beyond its
  own past" is the genius: prices have momentum/cycles, so a metric might *look*
  predictive just by drifting along with those cycles. Granger first lets the
  return predict itself, then asks if the metric adds *anything extra*. A fair,
  demanding bar. Computed with an **F-test** ("does adding these variables improve
  the fit enough to matter?").
- **A note on stationarity:** these tests need **stationary** data — series whose
  behavior doesn't drift over time. Raw prices drift; daily *returns* don't. So we
  used returns as the target and "differenced" the features (looked at day-to-day
  *changes*) to make them well-behaved.
- **Found:** **not one on-chain metric passed**, for either coin. The only thing
  that helped predict price was *past price itself* (a momentum feature). This is
  the strongest single piece of evidence that on-chain adds nothing.

### Test 3 — Mutual information (+ the clock trap)
- **What:** correlation only catches straight-line links. **Mutual information (MI)**
  detects *any* dependence — even curved or threshold ones ("only matters when very
  high"). `0` = truly unrelated; higher = more connected. A non-linear safety net.
- **The trap — the calendar in disguise:** MI flagged features like `SplyCur`
  (ADA's coin supply). But supply only ever goes *up*, so its value basically tells
  you *what date it is* (measured correlation with time ≈ 0.998). And returns
  cluster by era (2021 mostly up, 2022 mostly down). So MI wasn't finding "predicts
  returns" — it was finding "this metric knows the year, and years have moods."
  That's a **confound**, not a usable signal.
- **How we exposed it:** added a "correlation-with-time" column, making the
  high-MI features visibly just clocks.

---

## Why doing all three (plus the ablation) matters

Any single test can be fooled — we saw each one's blind spot. But they have
**different** blind spots. So when:

- the **ablation** (controls for price) finds nothing,
- **Granger** (controls for the past) finds nothing,
- **lead-lag with Newey-West** (controls for overlap) finds nothing real,
- and **MI** (controls for nothing, and gets fooled by the calendar) only "finds"
  clocks,

…they **converge** on the same answer from four independent directions. That
convergence is what upgrades a wishy-washy *"my model didn't work"* into a
confident, defensible conclusion:

> **On-chain metrics carry no real daily directional signal for BTC/ADA over
> 2020–2026 — the only thing that predicts price is price itself.**

---

## One-line glossary (for quick recall)

| Term | One line |
|------|----------|
| **Correlation (Pearson/Spearman)** | Do two things move together? Pearson = straight-line, Spearman = same-rank (curve-friendly). |
| **p-value** | Chance of seeing a result this strong if nothing real were going on. Small = probably real. |
| **F1** | 0–1 grade balancing false alarms vs misses for one class. |
| **MCC** | Stricter −1…+1 grade; 0 = guessing. Can't be faked by class imbalance. |
| **Weighted MCC** | MCC that penalizes opposite-direction (Buy↔Sell) errors 2×. |
| **Fold / walk-forward** | Train on past, test on the next chunk, march forward — never peek at the future. |
| **Std across folds** | How much the score jumps fold to fold. Big std = unstable = untrustworthy average. |
| **Permutation/shuffle test** | Shuffle labels to build a "pure luck" pile; see if the real score beats it. |
| **Wilcoxon signed-rank** | Paired test: does B beat A consistently across folds, or just wobble around 0? |
| **Bonferroni** | Strict multiple-testing fix: threshold ÷ number of tests. |
| **FDR / Benjamini–Hochberg** | Smarter multiple-testing fix: limits the fraction of false discoveries. |
| **Ablation** | A/B test: same model with vs without on-chain features; the gap = on-chain's true contribution. |
| **Overlapping returns** | Multi-day forward returns of nearby days share most of their window → not independent. |
| **Newey-West** | Corrected standard error/p-value that accounts for that overlap (deflates fake confidence). |
| **Granger causality** | Does X's past predict Y beyond Y's own past? The fair "extra info" test. |
| **Stationarity / differencing** | Series that don't drift over time; use returns / day-to-day changes to achieve it. |
| **Mutual information (MI)** | Detects *any* dependence, linear or not. 0 = independent. |
| **Trend/clock confound** | A steadily-rising feature secretly encodes the date; looks predictive but isn't. |
