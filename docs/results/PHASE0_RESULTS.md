# PHASE0_RESULTS.md — trustworthy baseline numbers

Phase 0 of `VALIDATION_PLAN.md`: make the existing LR/XGBoost numbers reproducible
and honest before any new validation. **No modelling conclusions are drawn here**
beyond "is the current signal trustworthy / stable" — the real tests (significance,
backtest, ablation) come in later phases.

Run date: 2026-06-22 · pipeline re-executed end-to-end (`2a`, `2b`) · seeds pinned (`random_state=42`).

---

## What was done

1. **Re-executed `2a` and `2b` end-to-end** from the existing raw snapshot (notebook `1`
   was *not* re-run, to avoid changing the data mid-validation / hitting the live
   Coin Metrics + Blockfrost APIs). Engineered files were regenerated from raw for
   code↔data consistency.
2. **Recorded the data snapshot:** BTC & ADA, **2020-01-01 → 2026-06-01**, 2344 rows/asset
   (1949 usable at BTC 30d/1% after `dropna`).
3. **Confirmed reproducibility:** seeds were already pinned; re-runs reproduce identical
   means (deterministic).
4. **Added "never one number" reporting:** per-fold std on the 24-config F1 table plus a
   new per-fold stability cell (mean ± std across walk-forward folds) in each notebook.
5. **Refreshed stale markdown** ("Final Analysis" in both) and removed references to a
   non-existent `4_LSTM` notebook.
6. **Repaired notebook structure:** both notebooks were `nbformat_minor=0` (cells had no
   IDs), which had caused earlier in-place edits to mis-target cells. Rebuilt cell order,
   bumped to `nbformat_minor=5`, assigned stable IDs.

---

## Headline numbers (BTC 30d / 1% threshold)

| Metric | LR | XGBoost |
|--------|----|---------|
| Buy MCC | 0.0476 | 0.0914 |
| Sell MCC | 0.0574 | 0.0993 |
| Buy **weighted** MCC | −0.1726 | −0.2238 |
| Sell **weighted** MCC | −0.1541 | −0.2176 |
| Hold predicted | 40.6% | 0.8% |

### 24-configuration F1 (mean ± spread across configs)

| Metric | LR | XGBoost | Delta |
|--------|----|---------|-------|
| Buy F1 | 0.214 ± 0.081 | 0.373 ± 0.071 | +74% |
| Sell F1 | 0.246 ± 0.104 | 0.435 ± 0.126 | +77% |

Best single config (best-of-24, **not** a confirmed edge): **ADA 7d / 0.5% → Sell F1 0.620**.

### Per-fold stability — the key Phase 0 finding (BTC 30d/1%, 26 folds)

| Metric | LR (mean ± std) | XGBoost (mean ± std) | Folds defined |
|--------|-----------------|----------------------|---------------|
| Buy F1  | 0.296 ± 0.297 | 0.467 ± 0.302 | 26/26 |
| Sell F1 | 0.186 ± 0.261 | 0.343 ± 0.316 | 26/26 |
| Buy MCC  | 0.010 ± 0.230 | 0.242 ± 0.202 | LR 15 / XGB 15 of 26 |
| Sell MCC | 0.092 ± 0.196 | 0.247 ± 0.197 | LR 12 / XGB 15 of 26 |

---

## Honest read (Phase 0 only)

- **The std is as large as the mean** for every per-fold metric. Fold-to-fold the signal
  is statistically indistinguishable from noise; a single pooled number overstates it.
- **MCC is undefined in ~40–50% of folds** because at 1%/30d, Hold is only ~6% of labels
  and Buy or Sell is often absent from a 30-day test window. The "3-class" task is
  effectively Buy-vs-Sell.
- **Weighted MCC is negative for both models.** Penalising opposite-direction errors 2×,
  the signal is net harmful — XGBoost more so, because it commits (Hold 0.8%) and makes
  319 wrong-direction calls vs LR hedging into Hold.
- **XGBoost is the better classifier** (F1 +74–77%, per-fold MCC ~0.24 vs LR ~0.01–0.09)
  and is the right base for further work — but better ≠ tradeable.

**Conclusion:** on this evidence, neither model is a tradeable directional signal. This is
a legitimate (weak/negative) result; confirming it rigorously requires Phases 1–2.

---

## Caveats / open risks carried into later phases

- These are **statistical** metrics, not money. No economic backtest yet (Phase 1.2).
- No significance test yet — is MCC 0.09 different from 0? (Phase 1.1).
- ADA's signal is dominated by Blockfrost staking/supply features; the **epoch→daily
  expansion must be audited for look-ahead** before ADA results are trusted (Phase 3.3).
- 30d-horizon labels overlap between neighbouring rows → consider purged/embargo CV
  (Phase 3.4).

## Reproduce

```powershell
.\.venv\Scripts\Activate.ps1
python -m jupyter nbconvert --to notebook --execute --inplace `
  --ExecutePreprocessor.kernel_name=python3 pipeline\2a_lr_model_pipeline.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace `
  --ExecutePreprocessor.kernel_name=python3 pipeline\2b_xgboost_model_pipeline.ipynb
```

(LR ~15 min, XGBoost ~40 min on this machine. Numbers are deterministic.)
