"""Step 3: causal, monthly-rebalanced fixed-weight portfolio benchmarks.
Run from project root: python -m src.step3_portfolios
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.step2_eda import ROOT, EXPECTED, TRADING_DAYS

WEIGHTS = {
    'Equal_weight': {'SPY': .125, 'QQQ': .125, 'IWM': .125, 'EFA': .125, 'EEM': .125, 'AGG': .125, 'GLD': .125, 'VNQ': .125},
    'Equity_focused': {'SPY': .40, 'QQQ': .30, 'IWM': .15, 'EFA': .15},
    'Balanced': {'SPY': .40, 'AGG': .30, 'GLD': .20, 'EFA': .10},
    'Defensive': {'AGG': .50, 'GLD': .30, 'SPY': .20},
    'SPY_buy_hold': {'SPY': 1.0},
}
SPLITS = {'train': ('2015-01-01','2022-12-31'), 'validation': ('2023-01-01','2024-12-31'), 'test': ('2025-01-01','2025-12-31')}


def load_returns(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Missing Step 2 returns: {path}. Run python -m src.step2_eda first.')
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='raise')
    if df.index.tz is not None or df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError('Return dates must be sorted, unique and timezone naive')
    if df.empty or len(df) < 2 or set(df.columns) != set(EXPECTED) or df.columns.duplicated().any():
        raise ValueError('Invalid return dataset or ETF columns')
    df = df[EXPECTED].astype(float)
    if not np.isfinite(df.to_numpy()).all() or (df <= -1).any().any():
        raise ValueError('Returns must be finite and greater than -100%')
    return df


def target_weights(alloc: dict[str,float], columns: pd.Index) -> np.ndarray:
    if any(k not in columns for k in alloc):
        raise ValueError('Unknown ticker in allocation')
    w = np.array([alloc.get(c, 0.0) for c in columns], dtype=float)
    if not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1.0, atol=1e-10):
        raise ValueError('Weights must be nonnegative and sum to one')
    return w


def simulate(returns: pd.DataFrame, weights: np.ndarray, cost_bps: float = 10.0,
             rebalance: str = 'monthly', initial_wealth: float = 10000.0) -> pd.DataFrame:
    """Trade at preceding close; cost paid before next close-to-close return.

    First return date is initial allocation date (initial investment is charged a fee).
    Each new month triggers a rebalance BEFORE that day's return is earned.
    No prices or returns from future dates are used to set trades.
    """
    if cost_bps < 0 or not np.isfinite(cost_bps) or cost_bps >= 10000:
        raise ValueError('cost_bps must be between 0 and 10000')
    if initial_wealth <= 0 or not np.isfinite(initial_wealth):
        raise ValueError('Initial wealth must be positive')
    if rebalance not in ('monthly','none'):
        raise ValueError('rebalance must be monthly or none')
    arr = returns.to_numpy(dtype=float)
    if not np.isfinite(arr).all() or (arr <= -1).any():
        raise ValueError('Invalid returns')
    if len(weights) != arr.shape[1] or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError('Invalid weights')
    drifted = np.zeros_like(weights, dtype=float)  # before initial purchase: all cash
    net_wealth = float(initial_wealth)
    gross_wealth = float(initial_wealth)
    gross_weights = weights.copy()
    rows = []
    prev_month = None
    for i, (date, daily) in enumerate(zip(returns.index, arr)):
        month = date.to_period('M')
        trade = (i == 0) or (rebalance == 'monthly' and month != prev_month)
        if trade:
            turnover = float(np.abs(weights - drifted).sum())
            drifted = weights.copy()
            gross_weights = weights.copy()
        else:
            turnover = 0.0
        fee_fraction = turnover * cost_bps / 10000.0
        gross_r = float(gross_weights @ daily)
        net_r_before_fee = float(drifted @ daily)
        net_r = (1.0 - fee_fraction) * (1.0 + net_r_before_fee) - 1.0
        net_wealth *= (1 + net_r)
        gross_wealth *= (1 + gross_r)
        drifted = drifted * (1 + daily) / (1 + net_r_before_fee)
        gross_weights = gross_weights * (1 + daily) / (1 + gross_r)
        rows.append((date, net_r, gross_r, net_wealth, gross_wealth, turnover, fee_fraction, bool(trade)))
        prev_month = month
    return pd.DataFrame(rows, columns=['date','net_return','gross_return','net_wealth','gross_wealth',
                                        'turnover','cost_fraction','rebalanced']).set_index('date')


def metrics(r: pd.Series, annual_rf: float = .02) -> dict:
    if r.empty:
        raise ValueError('No returns to evaluate')
    rf_daily = (1 + annual_rf)**(1/TRADING_DAYS) - 1
    wealth = (1 + r).cumprod()
    # Include starting wealth 1 for a correct first-day drawdown.
    path = np.r_[1.0, wealth.to_numpy()]
    dd = path / np.maximum.accumulate(path) - 1
    n = len(r)
    cagr = float(wealth.iloc[-1]**(TRADING_DAYS/n)-1)
    vol = float(r.std(ddof=1)*np.sqrt(TRADING_DAYS)) if n > 1 else float('nan')
    excess = r.to_numpy() - rf_daily
    downside = float(np.sqrt(np.mean(np.minimum(excess,0)**2))*np.sqrt(TRADING_DAYS))
    sharpe = float(excess.mean()*TRADING_DAYS/vol) if vol > 0 else float('nan')
    sortino = float(excess.mean()*TRADING_DAYS/downside) if downside > 0 else float('nan')
    return {'observations':n, 'total_return':float(wealth.iloc[-1]-1), 'cagr':cagr,
            'annualized_volatility':vol, 'sharpe_ratio':sharpe, 'sortino_ratio':sortino,
            'max_drawdown':float(dd.min())}


def run(path: Path, out: Path, cost_bps: float, annual_rf: float, initial_wealth: float) -> None:
    if annual_rf <= -1:
        raise ValueError('Risk-free rate must exceed -100%')
    returns = load_returns(path)
    # A separate continuous simulation per period resets wealth/weights at the period boundary.
    # This avoids using prior-period drift when reporting standalone split results.
    results = []
    daily_records = []
    turnover_records = []
    for split, (start, end) in SPLITS.items():
        subset = returns.loc[start:end]
        if len(subset) < 2:
            raise ValueError(f'Missing sufficient {split} data between {start} and {end}')
        for name, alloc in WEIGHTS.items():
            w = target_weights(alloc, subset.columns)
            rebalance = 'none' if name == 'SPY_buy_hold' else 'monthly'
            sim = simulate(subset, w, cost_bps, rebalance, initial_wealth)
            stats = metrics(sim['net_return'], annual_rf)
            stats.update({'split':split, 'portfolio':name, 'total_turnover':float(sim.turnover.sum()),
                          'rebalance_events':int(sim.rebalanced.sum()),
                          'total_cost_fraction_of_traded_notional':float(sim.cost_fraction.sum()),
                          'final_wealth_usd':float(sim.net_wealth.iloc[-1])})
            results.append(stats)
            tmp = sim.reset_index()
            tmp.insert(1, 'split', split)
            tmp.insert(2, 'portfolio', name)
            daily_records.append(tmp)
            turnover_records.append({'split':split,'portfolio':name,'turnover':float(sim.turnover.sum())})
    out.mkdir(parents=True, exist_ok=True)
    charts = out/'charts'
    charts.mkdir(exist_ok=True)
    summary = pd.DataFrame(results)
    daily = pd.concat(daily_records, ignore_index=True)
    summary.to_csv(out/'portfolio_metrics_by_split.csv', index=False)
    daily.to_csv(out/'portfolio_daily_paths.csv', index=False)
    daily.to_parquet(out/'portfolio_daily_paths.parquet', index=False)
    pd.DataFrame([{'portfolio':name, **{ticker:alloc.get(ticker,0) for ticker in EXPECTED}} for name,alloc in WEIGHTS.items()]).to_csv(out/'target_allocations.csv', index=False)
    for split in SPLITS:
        d = daily[daily.split == split]
        fig, ax = plt.subplots(figsize=(11,6))
        for name in WEIGHTS:
            s = d[d.portfolio == name]
            ax.plot(s.date, s.net_wealth, label=name.replace('_',' '))
        ax.set(title=f'{split.title()} — net portfolio wealth', ylabel='Portfolio value (USD)', xlabel='Date')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(charts/f'{split}_wealth.png', dpi=160)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(11,5))
        for name in WEIGHTS:
            s = d[d.portfolio == name]
            w = np.r_[initial_wealth, s.net_wealth.to_numpy()]
            dd = w/np.maximum.accumulate(w)-1
            ax.plot(s.date, dd[1:]*100, label=name.replace('_',' '))
        ax.set(title=f'{split.title()} — net drawdowns', ylabel='Drawdown (%)', xlabel='Date')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(charts/f'{split}_drawdown.png', dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10,5))
    pivot = summary.pivot(index='portfolio',columns='split',values='sharpe_ratio')
    pivot = pivot.reindex(list(WEIGHTS))
    pivot.plot(kind='bar',ax=ax)
    ax.set(title='Sharpe ratio by evaluation period',xlabel='',ylabel='Sharpe ratio')
    ax.tick_params(axis='x',rotation=25)
    fig.tight_layout()
    fig.savefig(charts/'sharpe_by_split.png',dpi=160)
    plt.close(fig)
    report = {'input':str(path),'periods':SPLITS,'initial_wealth_usd':initial_wealth,
              'transaction_cost_bps_per_dollar_traded':cost_bps,'annual_risk_free_rate_assumed':annual_rf,
              'fee_model':'sum(abs(target - pretrade weights)) * cost_bps / 10000; initial purchase charged',
              'rebalancing':'before first close-to-close return of each new month, using preceding close',
              'split_policy':'independent simulations reset to initial wealth at each split start',
              'warnings':['2015-2025 EDA already inspected: 2025 is not a pristine unseen holdout',
                          'Adjusted ETF data are not guaranteed point-in-time investable',
                          'No taxes, spreads, slippage, financing, or FX conversion modelled',
                          'Annualisation uses 252 observations/year; split CAGR is not calendar-day CAGR']}
    (out/'methodology.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\n=== STEP 3: NET PORTFOLIO PERFORMANCE ===')
    display = summary[['split','portfolio','cagr','annualized_volatility','sharpe_ratio','sortino_ratio','max_drawdown','final_wealth_usd']].copy()
    for col in ('cagr','annualized_volatility','max_drawdown'):
        display[col] = (display[col]*100).round(2)
    print(display.round(3).to_string(index=False))
    print(f'\nSaved Step 3 reports and charts to {out}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--returns',type=Path,default=ROOT/'data/processed/aligned_daily_returns_2015_2025.parquet')
    parser.add_argument('--out',type=Path,default=ROOT/'reports/step3')
    parser.add_argument('--cost-bps',type=float,default=10.0)
    parser.add_argument('--risk-free-rate',type=float,default=.02)
    parser.add_argument('--initial-wealth',type=float,default=10000.0)
    args = parser.parse_args()
    run(args.returns,args.out,args.cost_bps,args.risk_free_rate,args.initial_wealth)

if __name__ == '__main__':
    main()
