"""Momentum backtest - quantifying the one effect that passed Granger.

Across the whole project, the only feature that Granger-caused next-day returns was
**price momentum** (`Price_vs_MA7`), for BTC. This backtests the simplest rule built
from it - *hold when price is above its moving average, else sit in cash* - to see
whether that statistically-real effect is economically useful.

NOTE: this is a PRICE strategy, not on-chain. It is the "what does work" counterpoint
to the on-chain nulls; it does not revive the on-chain thesis. Momentum is a well-known
effect - here we simply quantify it on BTC/ADA.

Focus of this run: **how often does it win, before fees** (hit rates), plus a fee
sensitivity sweep so you can see when the edge dies. Fixed rule (no parameter fitting),
position decided at close t and acted at t+1 -> no look-ahead.

Run:  python utils/momentum_backtest.py
Writes: data/momentum_backtest.csv, docs/results/MOMENTUM_BACKTEST_RESULTS.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.feature_engineering import load_engineered_frames  # noqa: E402

MA_WINDOWS = (7, 14, 30, 50)      # MA7 is the Granger-significant one
FEES = (0.0, 0.001, 0.0025)       # 0 bps (headline), 10 bps, 25 bps
ANN = 365.0


def _trades(pos: np.ndarray, ret: np.ndarray):
    """Cumulative return of each contiguous long episode (pos==1 run)."""
    out = []
    i, n = 0, len(pos)
    while i < n:
        if pos[i] == 1:
            j = i
            while j < n and pos[j] == 1:
                j += 1
            out.append(np.prod(1 + ret[i:j]) - 1.0)
            i = j
        else:
            i += 1
    return np.array(out)


def _metrics(strat: pd.Series, pos: pd.Series, ret: pd.Series):
    eq = (1 + strat).cumprod()
    days = len(strat)
    cagr = eq.iloc[-1] ** (ANN / days) - 1 if days and eq.iloc[-1] > 0 else np.nan
    sharpe = strat.mean() / strat.std() * np.sqrt(ANN) if strat.std() > 0 else np.nan
    downside = strat[strat < 0].std()
    sortino = strat.mean() / downside * np.sqrt(ANN) if downside and downside > 0 else np.nan
    maxdd = float((eq / eq.cummax() - 1).min())
    active = strat[pos == 1]
    daily_hit = float((active > 0).mean()) if len(active) else np.nan
    monthly = strat.resample("ME").apply(lambda r: (1 + r).prod() - 1)
    monthly_hit = float((monthly > 0).mean()) if len(monthly) else np.nan
    tr = _trades(pos.to_numpy(), ret.to_numpy())
    trade_win = float((tr > 0).mean()) if len(tr) else np.nan
    return {
        "cagr": cagr, "sharpe": sharpe, "sortino": sortino, "maxdd": maxdd,
        "time_in_mkt": float((pos == 1).mean()), "n_trades": int(len(tr)),
        "trade_win_rate": trade_win, "daily_hit_rate": daily_hit,
        "monthly_hit_rate": monthly_hit,
    }


def backtest(price: pd.Series, ma: int, fee: float):
    ret = price.pct_change().fillna(0.0)
    signal = (price > price.rolling(ma).mean()).astype(float)   # decided at close t
    pos = signal.shift(1).fillna(0.0)                            # act next day
    trades = pos.diff().abs().fillna(0.0)
    strat = pos * ret - trades * fee
    valid = price.rolling(ma).mean().shift(1).notna()
    return _metrics(strat[valid], pos[valid], ret[valid])


def buy_hold(price: pd.Series):
    ret = price.pct_change().fillna(0.0)
    eq = (1 + ret).cumprod()
    monthly = ret.resample("ME").apply(lambda r: (1 + r).prod() - 1)
    return {
        "cagr": eq.iloc[-1] ** (ANN / len(ret)) - 1,
        "sharpe": ret.mean() / ret.std() * np.sqrt(ANN),
        "maxdd": float((eq / eq.cummax() - 1).min()),
        "daily_hit_rate": float((ret > 0).mean()),
        "monthly_hit_rate": float((monthly > 0).mean()),
    }


def main():
    btc, ada = load_engineered_frames()
    rows = []
    bh = {}
    for asset, df in [("BTC", btc), ("ADA", ada)]:
        price = df["PriceUSD"]
        bh[asset] = buy_hold(price)
        for ma in MA_WINDOWS:
            for fee in FEES:
                m = backtest(price, ma, fee)
                m.update({"asset": asset, "ma": ma, "fee_bps": int(fee * 1e4)})
                rows.append(m)
        print(f"[momentum] {asset} done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(_ROOT / "data" / "momentum_backtest.csv", index=False)
    print("Saved data/momentum_backtest.csv")
    for _, r in res[res.fee_bps == 0].sort_values(["asset", "ma"]).iterrows():
        print(f"  {r.asset} MA{int(r.ma):>2} 0bps: trade-win {r.trade_win_rate:.0%} "
              f"daily {r.daily_hit_rate:.0%} monthly {r.monthly_hit_rate:.0%} "
              f"CAGR {r.cagr:+.0%} Sharpe {r.sharpe:+.2f}", flush=True)
    _write_markdown(res, bh)


def _write_markdown(res: pd.DataFrame, bh: dict):
    md = _ROOT / "docs" / "results" / "MOMENTUM_BACKTEST_RESULTS.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    L = [
        "# MOMENTUM_BACKTEST_RESULTS.md - quantifying the one significant effect",
        "",
        "The only feature that Granger-caused next-day returns anywhere in this project was",
        "**price momentum** (`Price_vs_MA7`, BTC). This backtests the simplest rule from it:",
        "**hold the asset when price > its N-day moving average, else cash** (long/flat, fixed",
        "rule, acted next day - no look-ahead, no parameter fitting).",
        "",
        "> This is a **price** strategy, not on-chain - the 'what does work' counterpoint to",
        "> the on-chain nulls. Momentum is a well-known effect; we are only quantifying it.",
        "",
        "Focus: **how often it wins, before fees**, then a fee sweep (0 / 10 / 25 bps). MA7 is",
        "the Granger-significant window; 14/30/50 are robustness checks.",
        "",
        "## Buy-and-hold reference (whole sample)",
        "",
        "| Asset | CAGR | Sharpe | MaxDD | daily positive % | monthly positive % |",
        "|-------|------|--------|-------|------------------|--------------------|",
    ]
    for a, b in bh.items():
        L.append(f"| {a} | {b['cagr']:+.1%} | {b['sharpe']:+.2f} | {b['maxdd']:+.1%} | "
                 f"{b['daily_hit_rate']:.1%} | {b['monthly_hit_rate']:.1%} |")

    zero = res[res.fee_bps == 0]
    wr_lo, wr_hi = zero.trade_win_rate.min(), zero.trade_win_rate.max()
    btc7 = res[(res.asset == "BTC") & (res.ma == 7)].set_index("fee_bps")
    L += ["",
          "## Bottom line (answering: how often does it win, before fees?)",
          "",
          f"- **It wins *infrequently*.** Individual trades end positive only "
          f"**{wr_lo:.0%}-{wr_hi:.0%}** of the time; daily hit-rate ~50% and monthly ~30-50% - "
          "no better than just holding. Momentum makes money (when it does) via **asymmetry** "
          "(a few big winners, many small losers) and by **cutting downtrends**, not by being "
          "right often.",
          f"- **The Granger-significant window (MA7) is the weakest.** Even at 0 bps it "
          f"UNDERPERFORMS buy-and-hold on BTC ({btc7.loc[0,'cagr']:+.0%} vs {bh['BTC']['cagr']:+.0%} "
          f"CAGR), and with realistic fees it collapses ({btc7.loc[25,'cagr']:+.0%} at 25 bps). "
          "Statistical significance did NOT translate into a tradeable edge at that frequency.",
          "- **Longer windows (MA30/50) look better risk-adjusted** (higher Sharpe, shallower "
          "drawdown than buy-and-hold), and ADA momentum beats buy-and-hold broadly - **but** "
          "this is in-sample over ~1.5 cycles, and MA30/50 were not the Granger-tested windows "
          "(picking them is mild hindsight/multiple-testing).",
          "- **Honest verdict:** momentum's value here is **drawdown / risk reduction**, not a "
          "high win-rate and not out-of-sample-proven excess return. It is a *price* effect - the "
          "'what does work (a bit)' counterpoint to the on-chain nulls, not an on-chain result.",
          "",
          "## Momentum strategy - BEFORE FEES (0 bps): how often does it win?",
          "",
          "| Asset | MA | trade win-rate | daily hit-rate | monthly hit-rate | time in mkt | # trades | CAGR | Sharpe | MaxDD |",
          "|-------|----|----------------|----------------|------------------|-------------|----------|------|--------|-------|"]
    zero = res[res.fee_bps == 0].sort_values(["asset", "ma"])
    for _, r in zero.iterrows():
        L.append(f"| {r.asset} | {int(r.ma)} | {r.trade_win_rate:.1%} | {r.daily_hit_rate:.1%} | "
                 f"{r.monthly_hit_rate:.1%} | {r.time_in_mkt:.0%} | {int(r.n_trades)} | "
                 f"{r.cagr:+.1%} | {r.sharpe:+.2f} | {r.maxdd:+.1%} |")

    L += ["",
          "**How to read the win-rates:**",
          "- **trade win-rate** = of all long episodes, the % that ended net positive.",
          "- **daily hit-rate** = of days the strategy was in the market, the % that were up.",
          "- **monthly hit-rate** = % of calendar months with a positive strategy return.",
          "",
          "Momentum's signature is usually a **modest daily hit-rate (~50-55%) but positive",
          "expectancy** (winners bigger than losers) and a **better drawdown** than buy-and-hold,",
          "because it exits downtrends. Compare the win-rates and MaxDD to the buy-and-hold row.",
          "",
          "## Fee sensitivity - when does the edge die? (MA7)",
          "",
          "| Asset | fee (bps) | CAGR | Sharpe | MaxDD | # trades |",
          "|-------|-----------|------|--------|-------|----------|"]
    ma7 = res[res.ma == 7].sort_values(["asset", "fee_bps"])
    for _, r in ma7.iterrows():
        L.append(f"| {r.asset} | {int(r.fee_bps)} | {r.cagr:+.1%} | {r.sharpe:+.2f} | "
                 f"{r.maxdd:+.1%} | {int(r.n_trades)} |")
    L += ["",
          "Each switch costs the fee; MA7 trades often, so fees bite. If CAGR/Sharpe fall below",
          "buy-and-hold once realistic fees (10-25 bps) are applied, the statistically-real",
          "momentum effect is **not economically tradeable** at this frequency - a common and",
          "honest finding.",
          ""]
    md.write_text("\n".join(L), encoding="utf-8")
    print(f"Saved {md}")


if __name__ == "__main__":
    main()
