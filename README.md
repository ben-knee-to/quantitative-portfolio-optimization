# Quantitative Portfolio Analytics & Trading Strategy Backtesting

Educational Python project: market-data acquisition, risk measurement, portfolio optimisation and out-of-sample strategy evaluation.

## Step 1 — Data acquisition and profiling

Python 3.11 recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.step1_market_data
python -m pytest -q
```

Downloads daily dividend/split-adjusted closing prices for SPY, QQQ, IWM, EFA, EEM, AGG, GLD and VNQ over 2015-01-01 through 2025-12-31. Files are written to `data/raw`, `data/processed` and `reports` (all gitignored). No missing prices are filled.

This first iteration uses USD-denominated assets only. Later stages will add a separate South African / ZAR universe with appropriate currency conversion and calendar alignment. We will reserve 2026 for final forward-style evaluation and use chronological training/validation splits within 2015–2025.

**Data limitations:** Yahoo Finance data may be incomplete or revised. `yfinance` is an unofficial tool; review Yahoo's data-use terms. Do not redistribute raw downloaded datasets in this repository. Adjusted close is suitable for approximate total-return research but not an execution price or perfect total-return series. Survivorship bias remains because today's chosen funds are used historically.

## Roadmap
1. Acquire and validate market data
2. EDA: returns, drawdowns, volatility, correlations
3. Baseline portfolios and risk metrics
4. Portfolio optimisation
5. Monte Carlo, VaR, expected shortfall
6. Systematic strategy design
7. Cost-aware walk-forward backtesting
8. Optional ML return/risk forecasts
9. Streamlit dashboard and reproducibility audit

Not investment advice.
