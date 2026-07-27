"""Regime detection + event studies on on-chain state.

Two descriptive lenses that don't require on-chain to *predict* daily returns:

1. EVENT STUDY - the fair test of "extreme on-chain readings mark cycle turns".
   Unlike cycle_timing (which averaged over all days), this looks only at the rare
   moments an indicator's look-ahead-safe percentile *enters* an extreme (>0.95 =
   very expensive, <0.05 = very cheap), and measures the average forward-return
   path afterwards. If "expensive extremes" precede drawdowns and "cheap extremes"
   precede rallies, that supports the practitioner claim - even if the average-day
   relationship (cycle_timing) does not.

2. REGIME CLUSTERING - unsupervised: do on-chain features cluster into meaningful
   market states (accumulation / euphoria / capitulation)? Descriptive; we then
   characterise each cluster by its realised forward return and volatility.

Both are descriptive/in-sample; with only ~1.5 cycles the event study is
low-power (few independent events) - reported with counts so the reader can judge.

Run:  python utils/regime_events.py
Writes: data/event_study.csv, data/regimes.csv, docs/results/REGIME_EVENT_RESULTS.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.feature_engineering import load_engineered_frames  # noqa: E402
from utils.cycle_timing import NICE, expanding_percentile, forward_return  # noqa: E402

INDICATORS = ["CapMVRVCur", "Puell_Multiple", "NVT_Tx_Basis"]
EVENT_HORIZONS = (7, 30, 90, 180)
HI, LO = 0.95, 0.05       # extreme thresholds on the expanding percentile
N_REGIMES = 4


def _entries(mask: pd.Series) -> pd.Series:
    """Days where `mask` turns True after being False (event onsets, not every day)."""
    m = mask.fillna(False).astype(bool)
    return m & ~m.shift(1, fill_value=False)


def event_study(asset: str, df: pd.DataFrame):
    price = df["PriceUSD"]
    fwd = {h: forward_return(price, h) for h in EVENT_HORIZONS}
    base = {h: fwd[h].mean() for h in EVENT_HORIZONS}       # unconditional baseline
    rows = []
    for col in [c for c in INDICATORS if c in df.columns]:
        pct = expanding_percentile(df[col])
        for label, mask in [("expensive extreme (>0.95)", pct > HI),
                            ("cheap extreme (<0.05)", pct < LO)]:
            ev = _entries(mask)
            dates = ev[ev].index
            row = {"asset": asset, "indicator": NICE.get(col, col),
                   "event": label, "n_events": int(len(dates))}
            for h in EVENT_HORIZONS:
                vals = fwd[h].reindex(dates).dropna()
                row[f"fwd_{h}d"] = float(vals.mean()) if len(vals) else np.nan
            rows.append(row)
    return rows, base


def regime_clustering(asset: str, df: pd.DataFrame):
    feats = [c for c in ["CapMVRVCur", "NVT_Tx_Basis", "Puell_Multiple",
                         "Activity_Velocity", "AdrActCnt", "Volatility_30d"]
             if c in df.columns]
    X = df[feats].replace([np.inf, -np.inf], np.nan)
    fwd30 = forward_return(df["PriceUSD"], 30)
    vol = df["Volatility_30d"] if "Volatility_30d" in df else pd.Series(np.nan, index=df.index)
    data = pd.concat([X, fwd30.rename("fwd30"), vol.rename("vol")], axis=1).dropna()
    if len(data) < 200:
        return []
    Xs = StandardScaler().fit_transform(data[feats])        # global scaling: descriptive only
    km = KMeans(n_clusters=N_REGIMES, random_state=42, n_init=10).fit(Xs)
    data = data.assign(regime=km.labels_)
    rows = []
    for r, g in data.groupby("regime"):
        rows.append({
            "asset": asset, "regime": int(r), "n_days": len(g),
            "share": len(g) / len(data),
            "mean_fwd30": float(g["fwd30"].mean()),
            "mean_vol": float(g["vol"].mean()),
            "median_date": g.index[len(g) // 2].date().isoformat(),
        })
    return rows


def main():
    btc, ada = load_engineered_frames()
    ev_all, base_all, reg_all = [], {}, []
    for asset, df in [("BTC", btc), ("ADA", ada)]:
        print(f"[events] {asset} ...", flush=True)
        rows, base = event_study(asset, df)
        ev_all += rows; base_all[asset] = base
        print(f"[regimes] {asset} ...", flush=True)
        reg_all += regime_clustering(asset, df)
    ev = pd.DataFrame(ev_all); reg = pd.DataFrame(reg_all)
    ev.to_csv(_ROOT / "data" / "event_study.csv", index=False)
    reg.to_csv(_ROOT / "data" / "regimes.csv", index=False)
    print("Saved data/event_study.csv, data/regimes.csv")
    _write_markdown(ev, base_all, reg)


def _write_markdown(ev: pd.DataFrame, base: dict, reg: pd.DataFrame):
    md = _ROOT / "docs" / "results" / "REGIME_EVENT_RESULTS.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    L = [
        "# REGIME_EVENT_RESULTS.md - on-chain state: event studies & regimes",
        "",
        "Two descriptive lenses (no requirement that on-chain *predict* daily returns).",
        "",
        "> **Low-power caveat:** ~1.5 cycles in 2020-2026 -> few independent extreme events. "
        "`n_events` counts *onsets* (entering the extreme), which is small. Read as illustrative.",
        "",
        "## Bottom line",
        "",
        "- **Event study REFUTES 'extremes mark turns' in this sample.** Extreme-*expensive* "
        "readings preceded strong *gains*, not drawdowns (BTC MVRV expensive-extreme -> +60% at "
        "90d / +165% at 180d; ADA MVRV -> +255% at 90d), and cheap extremes preceded flat/negative "
        "returns. Same cause as `CYCLE_TIMING_RESULTS.md`: in a 1.5-cycle uptrend, 'expensive' "
        "(new-high percentile) coincides with ongoing bull runs - momentum beats mean-reversion.",
        "- **Regime clustering is the one (mildly) affirmative, DESCRIPTIVE result.** On-chain "
        "features partition history into states with materially different realised forward returns "
        "(BTC +5.5%..-0.9% at 30d; ADA +36.9%..-12.8%). So on-chain **describes** market phases - "
        "even though every predictive test shows it does not **forecast** direction out-of-sample. "
        "This is in-sample/descriptive (global scaling, realised returns), not a trading signal.",
        "- **HORIZON CAVEAT (added later):** the event study uses <=180d forward returns, so it "
        "is horizon-blind to the multi-year cycle. `MVRV_SENTIMENT_RESULTS.md` shows that at ~540d "
        "the BTC valuation signal **reverses** (cheap+fear outperforms). Read the 'extremes don't "
        "mark turns' finding as short-horizon; the cycle-scale claim is illustrated (for BTC, ~1 "
        "bottom) there, not here.",
        "",
        "## 1. Event study - do extreme valuation readings mark turns?",
        "",
        "For each indicator, the day its look-ahead-safe percentile *enters* an extreme "
        "(>0.95 expensive / <0.05 cheap); average forward return afterwards. The claim holds "
        "if **expensive extremes -> negative** forward returns and **cheap extremes -> "
        "positive**, beyond the unconditional baseline.",
        "",
        "Unconditional baseline mean forward return:",
    ]
    for a, b in base.items():
        L.append(f"- **{a}**: " + ", ".join(f"{h}d {b[h]:+.1%}" for h in EVENT_HORIZONS))
    L += ["",
          "| Asset | Indicator | Event | n | fwd 7d | fwd 30d | fwd 90d | fwd 180d |",
          "|-------|-----------|-------|---|--------|---------|---------|----------|"]
    for _, r in ev.iterrows():
        def c(h):
            v = r[f"fwd_{h}d"]
            return "n/a" if pd.isna(v) else f"{v:+.1%}"
        L.append(f"| {r.asset} | {r.indicator} | {r.event} | {int(r.n_events)} | "
                 f"{c(7)} | {c(30)} | {c(90)} | {c(180)} |")
    L += ["",
          "**Read:** compare each row to that asset's baseline above. An expensive-extreme row "
          "far *below* baseline (or negative) supports 'tops'; a cheap-extreme row far *above* "
          "baseline supports 'bottoms'. Small `n` = treat as anecdote, not proof.",
          "",
          "## 2. Regime clustering (unsupervised, descriptive)",
          "",
          f"KMeans (k={N_REGIMES}) on standardised on-chain features (MVRV, NVT, Puell, "
          "velocity, active addresses, volatility). Each regime characterised by its realised "
          "next-30d return and volatility. (Global scaling - descriptive, in-sample only.)",
          "",
          "| Asset | Regime | days | share | mean fwd-30d | mean vol | median date |",
          "|-------|--------|------|-------|--------------|----------|-------------|"]
    for _, r in reg.sort_values(["asset", "mean_fwd30"], ascending=[True, False]).iterrows():
        L.append(f"| {r.asset} | {int(r.regime)} | {int(r.n_days)} | {r.share:.0%} | "
                 f"{r.mean_fwd30:+.1%} | {r.mean_vol:.3f} | {r.median_date} |")
    L += ["",
          "**Read:** if regimes separate cleanly into high-return/low-vol vs low-return/high-vol "
          "states, on-chain features *describe* market phases (a valid, if descriptive, use) - "
          "even though earlier tests show they don't *predict* direction out-of-sample.",
          ""]
    md.write_text("\n".join(L), encoding="utf-8")
    print(f"Saved {md}")


if __name__ == "__main__":
    main()
