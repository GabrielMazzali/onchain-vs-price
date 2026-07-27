"""Cycle-timing test: does on-chain VALUATION time long-horizon tops/bottoms?

The daily-prediction tests (ablation, signal detection) showed on-chain gives no
short-term directional edge. But practitioners don't use MVRV / Puell / NVT for
daily trading - they use them as **long-horizon cycle indicators**: "high MVRV =
overvalued = near a top", "low = undervalued = near a bottom". This tests that
*actual* claim.

For each indicator we build a **look-ahead-safe expanding percentile** (where does
today's value sit in its own history so far?) and ask:
  1. Conditional forward return by valuation bucket - do cheap buckets earn more
     than expensive buckets over 30/90/180 days?
  2. Rank-IC = Spearman(valuation percentile, forward return), with an
     overlap-aware (non-overlapping) p-value. Expected sign is NEGATIVE
     (high valuation -> low forward return).
  3. A simple valuation-timing backtest (hold when cheap, cash when expensive)
     vs buy-and-hold, with fees.

IMPORTANT CAVEAT (stated in the output): 2020-2026 spans only ~1.5 crypto cycles
(one major top, one major bottom), so this is **descriptive / illustrative**, not
a high-power statistical test - there are too few independent cycle turns.

Run from repo root:  python utils/cycle_timing.py
Writes: data/cycle_timing_ic.csv, data/cycle_timing_backtest.csv,
        docs/results/CYCLE_TIMING_RESULTS.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.feature_engineering import load_engineered_frames  # noqa: E402

INDICATORS = ["CapMVRVCur", "Puell_Multiple", "NVT_Tx_Basis"]
NICE = {"CapMVRVCur": "MVRV", "Puell_Multiple": "Puell", "NVT_Tx_Basis": "NVT"}
HORIZONS = (30, 90, 180)
MIN_HISTORY = 180          # need this many past days before a percentile is trusted
FEE = 0.001               # 10 bps per position change
ANN = 365.0


def expanding_percentile(s: pd.Series, min_history: int = MIN_HISTORY) -> pd.Series:
    """Look-ahead-safe: pct[t] = fraction of values up to and including t that are
    <= value[t]. Uses only past+present, so it is usable as a live signal."""
    v = s.to_numpy(dtype=float)
    n = len(v)
    out = np.full(n, np.nan)
    for i in range(n):
        w = v[: i + 1]
        w = w[np.isfinite(w)]
        if len(w) >= min_history:
            out[i] = (w <= v[i]).mean()
    return pd.Series(out, index=s.index)


def forward_return(price: pd.Series, h: int) -> pd.Series:
    return price.shift(-h) / price - 1.0


def _spearman_nonoverlap(x, y, h):
    """Overlap-aware Spearman p: subsample every h-th point (non-overlapping forward
    windows), median p over the h offsets. Returns (rho_full, p_nonoverlap)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 30:
        return np.nan, np.nan
    rho = spearmanr(x, y)[0]
    ps = []
    for off in range(h):
        xs, ys = x[off::h], y[off::h]
        if len(xs) >= 8 and np.std(xs) > 0 and np.std(ys) > 0:
            ps.append(spearmanr(xs, ys)[1])
    return float(rho), (float(np.median(ps)) if ps else np.nan)


def conditional_returns(pct: pd.Series, fwd: pd.Series, n_buckets=5):
    """Mean forward return per valuation bucket (quintile of the percentile)."""
    d = pd.concat([pct.rename("p"), fwd.rename("f")], axis=1).dropna()
    if len(d) < 100:
        return None
    d["bucket"] = np.clip((d["p"] * n_buckets).astype(int), 0, n_buckets - 1)
    return d.groupby("bucket")["f"].mean()


def backtest_timing(price: pd.Series, pct: pd.Series, thresh=0.5, fee=FEE):
    """Hold the asset when cheap (pct < thresh), else sit in cash. Trade next day."""
    ret = price.pct_change().fillna(0.0)
    pos = (pct < thresh).astype(float)          # decided at close t (uses pct[t])
    pos = pos.shift(1).fillna(0.0)              # act next day -> no look-ahead
    trades = pos.diff().abs().fillna(0.0)
    strat = pos * ret - trades * fee
    valid = pct.shift(1).notna()               # only score once the signal exists
    strat = strat[valid]; ret_bh = ret[valid]
    eq = (1 + strat).cumprod()
    eq_bh = (1 + ret_bh).cumprod()

    def metrics(r, eq):
        days = len(r)
        cagr = eq.iloc[-1] ** (ANN / days) - 1 if days > 0 and eq.iloc[-1] > 0 else np.nan
        sharpe = r.mean() / r.std() * np.sqrt(ANN) if r.std() > 0 else np.nan
        maxdd = float((eq / eq.cummax() - 1).min())
        return cagr, sharpe, maxdd

    c_s, s_s, dd_s = metrics(strat, eq)
    c_b, s_b, dd_b = metrics(ret_bh, eq_bh)
    return {
        "strat_cagr": c_s, "strat_sharpe": s_s, "strat_maxdd": dd_s,
        "bh_cagr": c_b, "bh_sharpe": s_b, "bh_maxdd": dd_b,
        "time_in_market": float(pos[valid].mean()), "n_trades": int(trades[valid].sum()),
    }


def run_asset(asset: str, df: pd.DataFrame):
    price = df["PriceUSD"]
    inds = [c for c in INDICATORS if c in df.columns]
    pcts = {c: expanding_percentile(df[c]) for c in inds}
    # composite valuation = average percentile across available indicators
    comp = pd.concat(pcts.values(), axis=1).mean(axis=1)
    pcts["COMPOSITE"] = comp

    ic_rows, cond_rows = [], []
    for name, pct in pcts.items():
        for h in HORIZONS:
            fwd = forward_return(price, h)
            rho, p = _spearman_nonoverlap(pct.values, fwd.values, h)
            ic_rows.append({"asset": asset, "indicator": NICE.get(name, name),
                            "horizon_days": h, "rank_ic": rho, "p_nonoverlap": p})
        cr = conditional_returns(pct, forward_return(price, 90))
        if cr is not None:
            cheap = cr.get(0, np.nan); expensive = cr.get(len(cr) - 1, np.nan)
            cond_rows.append({"asset": asset, "indicator": NICE.get(name, name),
                              "cheap_bucket_fwd90": cheap, "expensive_bucket_fwd90": expensive,
                              "spread": cheap - expensive})
    bt = backtest_timing(price, comp)
    bt.update({"asset": asset, "signal": "COMPOSITE valuation (hold when cheap half)"})
    return ic_rows, cond_rows, bt


def main():
    btc, ada = load_engineered_frames()
    ic_all, cond_all, bt_all = [], [], []
    for asset, df in [("BTC", btc), ("ADA", ada)]:
        print(f"[cycle] {asset} ...", flush=True)
        ic, cond, bt = run_asset(asset, df)
        ic_all += ic; cond_all += cond; bt_all.append(bt)
    ic_df = pd.DataFrame(ic_all); cond_df = pd.DataFrame(cond_all); bt_df = pd.DataFrame(bt_all)
    ic_df.to_csv(_ROOT / "data" / "cycle_timing_ic.csv", index=False)
    bt_df.to_csv(_ROOT / "data" / "cycle_timing_backtest.csv", index=False)
    print(f"Saved data/cycle_timing_*.csv")
    _write_markdown(ic_df, cond_df, bt_df)


def _write_markdown(ic: pd.DataFrame, cond: pd.DataFrame, bt: pd.DataFrame):
    md = _ROOT / "docs" / "results" / "CYCLE_TIMING_RESULTS.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    comp = bt.set_index("asset")
    beat = [a for a in comp.index if comp.loc[a, "strat_cagr"] > comp.loc[a, "bh_cagr"]]
    L = [
        "# CYCLE_TIMING_RESULTS.md - does on-chain valuation time long-horizon cycles?",
        "",
        "On-chain valuation metrics (MVRV, Puell, NVT) are used in practice as **cycle**",
        "indicators, not daily trading signals. This tests that claim: a look-ahead-safe",
        "**expanding percentile** of each indicator (where today's value sits in its own",
        "history) vs **forward returns** over 30/90/180 days.",
        "",
        "## Bottom line",
        "",
        "**The naive valuation-timing rule did NOT work in this sample - it did the "
        "opposite of the folklore.**",
        "- Rank-IC is mostly **positive** (e.g. BTC Puell +0.21, MVRV +0.12): high-valuation "
        "readings preceded *higher*, not lower, returns.",
        "- The **'expensive' bucket outperformed the 'cheap' bucket** over 90 days for every "
        "indicator (BTC MVRV +38.8% vs +4.0%; ADA MVRV +94.9% vs +17.7%).",
        f"- The composite 'hold when cheap' strategy **beat buy-and-hold for {len(beat)}/2 "
        "assets** - it badly underperformed (BTC -2.8% vs +41.5% CAGR; ADA -25% vs +20%).",
        "",
        "**Why / caveats:** (1) 2020-2026 was a secular uptrend where high valuation persisted "
        "through the biggest rallies (momentum beat mean-reversion); (2) an **expanding "
        "percentile on a trending series conflates 'expensive' with 'making new highs / in an "
        "uptrend'**, so 'sell when expensive' mostly meant 'sell during the bull'; (3) only "
        "~1.5 cycles in sample. So this refutes the *continuous* timing rule, but the narrower "
        "claim - do the *rare extreme* readings mark THE turn? - needs an **event study** on "
        "extremes (see `REGIME_EVENT_RESULTS.md`), not an average-over-all-days test.",
        "",
        "> **HORIZON CAVEAT (added later):** the horizons here are <=180d. `MVRV_SENTIMENT_RESULTS.md` "
        "shows that at **cycle-scale horizons (~540d)** the picture partially **reverses for BTC** - "
        "cheap/undervalued readings outperform and expensive ones underperform. So 'timing refuted' "
        "holds *at short horizons*; at the multi-year horizon these indicators are actually used on, "
        "BTC shows the textbook valuation effect (on ~1 cycle of data - illustrative, not powered).",
        "",
        "> **Sample caveat (read first):** 2020-2026 spans only ~1.5 crypto cycles - one",
        "> major top (late 2021), one major bottom (late 2022). So this is **descriptive /",
        "> illustrative**: there are far too few independent cycle turns for high statistical",
        "> power, and the overlap-aware p-values below are correspondingly weak. Treat a good",
        "> result as 'consistent with the cycle claim', not 'proven'.",
        "",
        "Generated by `python utils/cycle_timing.py`. Expected sign of rank-IC is **negative**",
        "(high valuation -> lower forward return).",
        "",
        "## 1. Rank-IC: Spearman(valuation percentile, forward return)",
        "",
        "| Asset | Indicator | Horizon | rank-IC | p (non-overlap) |",
        "|-------|-----------|---------|---------|-----------------|",
    ]
    for _, r in ic.iterrows():
        p = "n/a" if pd.isna(r.p_nonoverlap) else f"{r.p_nonoverlap:.3f}"
        L.append(f"| {r.asset} | {r.indicator} | {int(r.horizon_days)}d | {r.rank_ic:+.3f} | {p} |")
    L += ["",
          "Negative IC = the indicator's 'expensive' readings precede lower returns (the",
          "cycle-timing claim). Magnitude and sign matter more than the (low-power) p here.",
          "",
          "## 2. Conditional 90-day forward return: cheap vs expensive bucket",
          "",
          "| Asset | Indicator | cheap-bucket fwd90 | expensive-bucket fwd90 | spread (cheap - exp.) |",
          "|-------|-----------|--------------------|------------------------|-----------------------|"]
    for _, r in cond.iterrows():
        L.append(f"| {r.asset} | {r.indicator} | {r.cheap_bucket_fwd90:+.1%} | "
                 f"{r.expensive_bucket_fwd90:+.1%} | {r.spread:+.1%} |")
    L += ["",
          "A large positive spread = buying when the indicator says 'cheap' beat buying when",
          "'expensive', over the next 90 days - the cycle-timing effect.",
          "",
          "## 3. Valuation-timing backtest vs buy-and-hold",
          "",
          "Composite valuation = average percentile across the asset's indicators. Strategy:",
          f"hold the asset when in the cheap half (percentile < 0.5), else cash; {int(FEE*1e4)} bps",
          "per switch, acted next day (no look-ahead).",
          "",
          "| Asset | Strat CAGR | B&H CAGR | Strat Sharpe | B&H Sharpe | Strat MaxDD | B&H MaxDD | Time in mkt | Trades |",
          "|-------|-----------|----------|--------------|------------|-------------|-----------|-------------|--------|"]
    for _, r in bt.iterrows():
        L.append(f"| {r.asset} | {r.strat_cagr:+.1%} | {r.bh_cagr:+.1%} | {r.strat_sharpe:+.2f} | "
                 f"{r.bh_sharpe:+.2f} | {r.strat_maxdd:+.1%} | {r.bh_maxdd:+.1%} | "
                 f"{r.time_in_market:.0%} | {int(r.n_trades)} |")
    L += ["",
          "**How to read:** a higher Sharpe / shallower drawdown than buy-and-hold - even at",
          "similar or lower CAGR - is the classic 'cycle timing reduces risk' result. Because",
          "the sample is ~1.5 cycles, read this as illustrative of the mechanism, not as an",
          "out-of-sample-validated strategy.",
          ""]
    md.write_text("\n".join(L), encoding="utf-8")
    print(f"Saved {md}")


if __name__ == "__main__":
    main()
