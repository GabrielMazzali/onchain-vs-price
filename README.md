# Can on-chain metrics predict Bitcoin and Cardano prices?

A research codebase (undergraduate thesis / TCC, XP Educação — Bacharelado em Ciência de Dados) testing whether **on-chain metrics** carry a directional trading edge for **Bitcoin (BTC)** and **Cardano (ADA)** at daily frequency, over and above what you can already get from **price alone**.

> **The answer is no.** Over 2020-01-01 → 2026-06-01, on-chain metrics show no detectable incremental directional signal for BTC/ADA; the only real (but small, untradeable-as-tested) effect is price momentum. The full, plain-language answer with the evidence chain is in **[`docs/CONCLUSIONS.md`](docs/CONCLUSIONS.md)** — start there.

This is a **rigorous negative result** backed by supervised models, a controlled ablation, model-free statistical tests, and a deep-learning capacity check — plus a reusable **methodology for evaluating crypto predictive claims** (see [`docs/CONFOUNDS.md`](docs/CONFOUNDS.md)).

There is no packaged application; the deliverables are notebooks, CSV/Parquet artifacts, and the write-ups under `docs/`.

---

## Quickstart

Requires **Python 3.12**.

```powershell
# 1. Environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt

# 2. Secrets (only needed to re-fetch ADA staking data)
copy .env.example .env                # then paste your Blockfrost project ID
```

The repo ships the raw Coin Metrics snapshots in `data/raw/`, so you can run everything **without any API key**. The `.env` / Blockfrost step is only needed if you re-run the data loader to refresh ADA staking from scratch.

```powershell
# 3. Run the pipeline notebooks in order (from repo root)
python -m jupyter nbconvert --to notebook --execute pipeline\1_data_loader.ipynb
python -m jupyter nbconvert --to notebook --execute pipeline\2a_lr_model_pipeline.ipynb
python -m jupyter nbconvert --to notebook --execute pipeline\2b_xgboost_model_pipeline.ipynb

# ...or batch-run both model notebooks in parallel:
python utils\run_models_parallel.py
```

The **LSTM stage** (`pipeline/4_LSTM_model_pipeline.ipynb`) needs two extra packages — uncomment `torch` / `poutyne` in `requirements.txt` (or `pip install torch poutyne`) before running it.

---

## Repository layout

```
IC/
├── pipeline/                          ← main numbered notebooks (run in order)
│   ├── 1_data_loader.ipynb            ← fetch Coin Metrics + Blockfrost → data/raw + engineered files
│   ├── 2a_lr_model_pipeline.ipynb     ← logistic regression, walk-forward
│   ├── 2b_xgboost_model_pipeline.ipynb← XGBoost, walk-forward
│   ├── 3_ablation.ipynb               ← price-only vs +on-chain (presents utils/ablation.py)
│   ├── 3_signal_detection.ipynb       ← model-free lead-lag / Granger / MI (presents utils/signal_detection.py)
│   └── 4_LSTM_model_pipeline.ipynb    ← LSTM + DLinear capacity check + persistence-illusion demo (needs torch/poutyne)
├── notebooks/eda/eda_analysis.ipynb   ← signal distributions, correlations, weekday effects
├── src/feature_engineering.py         ← ALL shared helpers: paths, feature engineering, loaders, train-only scaler
├── utils/                             ← standalone analysis scripts (each writes a docs/results/*.md)
├── data/
│   ├── raw/coinmetrics_{btc,ada}.csv  ← committed API snapshot
│   └── engineered/                    ← generated, gitignored, auto-rebuilt on demand
└── docs/                              ← research narrative & results (start at docs/CONCLUSIONS.md)
```

## Running the analyses

Each analysis is a standalone script under `utils/` that regenerates a results file under `docs/results/`. Run from the repo root:

| Command | Produces |
|---------|----------|
| `python utils/ablation.py` | `docs/results/ABLATION_RESULTS.md` — price-only vs +on-chain (~30 min; `--quick` for a smoke test) |
| `python utils/signal_detection.py` | `docs/results/SIGNAL_DETECTION_RESULTS.md` — lead-lag / Granger / mutual information |
| `python utils/volatility_test.py` | `docs/results/VOLATILITY_RESULTS.md` — on-chain vs price for forward volatility |
| `python utils/cycle_timing.py` | `docs/results/CYCLE_TIMING_RESULTS.md` — MVRV/Puell/NVT cycle-timing test |
| `python utils/regime_events.py` | `docs/results/REGIME_EVENT_RESULTS.md` — regime clustering + event studies |
| `python utils/momentum_backtest.py` | `docs/results/MOMENTUM_BACKTEST_RESULTS.md` — price-momentum backtest |
| `python utils/mvrv_sentiment.py` | `docs/results/MVRV_SENTIMENT_RESULTS.md` — MVRV × sentiment conditional returns |

## Data flow

`1_data_loader` pulls Coin Metrics (community endpoint, no key) into `data/raw/`, adds ADA staking from Blockfrost, then calls `engineer_features` + `save_engineered`. The model notebooks and analysis scripts call `load_engineered_frames`, which rebuilds the engineered CSV/Parquet from raw if the cache is missing. Canonical paths (`RAW_DIR`, `ENG_DIR`) live in `src/feature_engineering.py` — never duplicate them in a notebook.

## Anti-leakage protocol

The model notebooks enforce three rules (`src/feature_engineering.py` is intentionally scaler-free):

1. **Per-fold scaling.** `StandardScaler` is fit on each fold's training window only — never on the full frame. (XGBoost skips scaling; it's scale-invariant.)
2. **No synthetic labels.** `classify_signal` returns `NaN` for `NaN` forward returns, so the last rows are dropped by `dropna()` rather than mislabeled as Hold.
3. **Time-aware inner CV.** `RandomizedSearchCV` uses `TimeSeriesSplit`, not shuffled K-Fold.

## Where to read the findings

| I want… | Read |
|---------|------|
| **The answer**, plain, with the evidence chain | [`docs/CONCLUSIONS.md`](docs/CONCLUSIONS.md) |
| Every statistical method explained + feature data dictionary | [`docs/METHODS_EXPLAINED.md`](docs/METHODS_EXPLAINED.md) |
| The confounds / methodology chapter (how naive analysis lies) | [`docs/CONFOUNDS.md`](docs/CONFOUNDS.md) |
| The deep-learning stage explained | [`docs/LSTM_EXPLAINED.md`](docs/LSTM_EXPLAINED.md) |
| The phased research plan / process record | [`docs/VALIDATION_PLAN.md`](docs/VALIDATION_PLAN.md) |
| The raw evidence tables | [`docs/results/`](docs/results/) |

## Notes

- Coin Metrics API responses change over time; the committed `data/raw/*.csv` snapshot a specific fetch. Re-running `1_data_loader` will refresh them.
- `data/engineered/` is gitignored and auto-rebuilt on first load, so it may be absent after clone.
- Contributor / architecture detail (what to edit where) lives in [`CLAUDE.md`](CLAUDE.md).

## License

This repository is dual-licensed:

- **Code** (the `.py` files, notebook code cells, scripts) — **MIT**, see [`LICENSE`](LICENSE). Free to use, modify, and redistribute, keeping the copyright notice.
- **Research & written materials** (`docs/`, this README, results tables, notebook narrative, figures) — **CC BY 4.0**, see [`LICENSE-docs`](LICENSE-docs). Free to reuse and adapt **with attribution**.

If you use the findings or methodology, please cite: *Gabriel Mazzali Garcia (2026), "Can on-chain metrics predict Bitcoin and Cardano prices?", undergraduate thesis (TCC), XP Educação.*
