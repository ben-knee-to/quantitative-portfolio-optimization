# Quantitative Trading & Portfolio Optimisation

Educational financial data science project using eight USD-denominated ETFs: SPY, QQQ, IWM, EFA, EEM, AGG, GLD, VNQ.

## Windows / PyCharm setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.step1_market_data
python run_all.py
python -m pytest -q
streamlit run app/streamlit_app.py
```

All commands should run from the repository root. Historical data downloads require an internet connection.

## Stages

1. Download and validate 2015–2025 adjusted ETF prices.
2. Exploratory data analysis, returns, correlations and drawdowns.
3. Fixed-weight portfolio benchmarks, monthly rebalancing and 10bps costs.
4. Training-only constrained minimum-variance and max-Sharpe weights.
5. Historical risk and iid bootstrap scenarios (not forward forecasts).
6. Prior-close momentum rotation with 126-session lookback.
7. Lagged-feature logistic-regression direction baseline.
8. Consolidated evaluation; Streamlit dashboard.

## Evaluation integrity

- Train: 2015–2022; validation: 2023–2024; chronological test: 2025.
- The 2025 asset-level EDA has already been viewed. Therefore the 2025 test is **not a pristine unseen holdout**.
- Fixed portfolio allocations are predeclared; optimised weights and ML model are fitted on train only.
- Momentum is a fixed-rule exploratory strategy; it uses prior-close information and no tuning on 2025.
- The simulator applies a stylised 10bp fee to turnover, not actual brokerage quotes. No tax, FX, spreads, or market impact.
- Results are historical, not financial advice. Adjusted prices and current ETF selection introduce potential data limitations.
- The bootstrap assumes independent identically distributed returns and understates regime-change/tail risk.

## GitHub

```powershell
git add .
git commit -m "Complete portfolio analytics pipeline and dashboard"
git push origin main
```

Generated datasets and reports are excluded from version control by `.gitignore`. Review files before committing secrets.
