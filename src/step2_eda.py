"""Step 2: aligned ETF EDA. Run: python -m src.step2_eda"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'AGG', 'GLD', 'VNQ']
TRADING_DAYS = 252


def load_prices(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Missing Step 1 dataset: {path}. Run python -m src.step1_market_data first.')
    if path.suffix == '.parquet':
        prices = pd.read_parquet(path)
    else:
        prices = pd.read_csv(path, index_col='date', parse_dates=['date'])
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices.index = pd.to_datetime(prices.index, errors='raise')
    if prices.index.tz is not None:
        raise ValueError('Timezone-aware dates not expected')
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError('Duplicate or unsorted price dates')
    if prices.columns.duplicated().any():
        raise ValueError('Duplicate ticker columns')
    if set(prices.columns) != set(EXPECTED):
        raise ValueError(f'Expected {EXPECTED}, received {list(prices.columns)}')
    prices = prices[EXPECTED].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(prices.to_numpy(dtype=float)[~prices.isna().to_numpy()]).all():
        raise ValueError('Infinite prices detected')
    if (prices <= 0).any().any():
        raise ValueError('Nonpositive prices detected')
    if prices.empty:
        raise ValueError('Empty price data')
    return prices


def align_prices(prices: pd.DataFrame, min_days: int = 252) -> tuple[pd.DataFrame, dict]:
    aligned = prices.dropna(how='any')
    # Returns are calculated only on consecutive rows of the complete-case price calendar.
    # This treats each gap between retained dates as one observation interval.
    if len(aligned) < min_days + 1:
        raise ValueError(f'Only {len(aligned)} complete price dates; need {min_days + 1}')
    gap_days = aligned.index.to_series().diff().dt.days.dropna()
    report = {
        'raw_price_dates': len(prices),
        'complete_price_dates': len(aligned),
        'dropped_price_dates': len(prices) - len(aligned),
        'missing_prices_by_asset': {k: int(v) for k, v in prices.isna().sum().items()},
        'first_aligned_date': aligned.index[0].date().isoformat(),
        'last_aligned_date': aligned.index[-1].date().isoformat(),
        'calendar_gaps_over_5_days': int((gap_days > 5).sum()),
        'max_calendar_gap_days': int(gap_days.max()),
        'return_intervals': len(aligned) - 1,
        'return_convention': 'simple returns between consecutive complete-case price dates',
        'annualisation_convention': '252 aligned observations per year',
    }
    return aligned, report


def calculate_metrics(prices: pd.DataFrame, annual_rf: float = 0.02) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if annual_rf <= -1:
        raise ValueError('Annual risk-free rate must exceed -100%')
    returns = prices.pct_change(fill_method=None).iloc[1:]
    if returns.isna().any().any() or not np.isfinite(returns.to_numpy()).all():
        raise ValueError('Invalid aligned returns')
    n = len(returns)
    rf_daily = (1 + annual_rf) ** (1 / TRADING_DAYS) - 1
    wealth = (1 + returns).cumprod()
    wealth = pd.concat([pd.DataFrame(1., index=prices.index[:1], columns=prices.columns), wealth])
    wealth.index.name = 'date'
    drawdown = wealth.div(wealth.cummax()).sub(1)
    rows = []
    for ticker in prices:
        r = returns[ticker]
        total = float(prices[ticker].iloc[-1] / prices[ticker].iloc[0] - 1)
        cagr = (1 + total) ** (TRADING_DAYS / n) - 1
        vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
        excess = r - rf_daily
        downside = float(np.sqrt(np.mean(np.minimum(excess.to_numpy(), 0) ** 2)) * np.sqrt(TRADING_DAYS))
        sharpe = float(excess.mean() * TRADING_DAYS / vol) if vol > 0 else np.nan
        sortino = float(excess.mean() * TRADING_DAYS / downside) if downside > 0 else np.nan
        dd = float(drawdown[ticker].min())
        q05 = float(r.quantile(.05))
        tail = r[r <= q05]
        rows.append({
            'ticker': ticker, 'observations': n, 'total_return': total, 'cagr': cagr,
            'annualized_arithmetic_return': float(r.mean() * TRADING_DAYS),
            'annualized_volatility': vol, 'sharpe_ratio': sharpe,
            'sortino_ratio': sortino, 'max_drawdown': dd,
            'calmar_ratio': float(cagr / abs(dd)) if dd < 0 else np.nan,
            'daily_var_95_loss': float(max(0, -q05)),
            'daily_expected_shortfall_95_loss': float(max(0, -tail.mean())),
            'positive_day_fraction': float((r > 0).mean()),
        })
    return pd.DataFrame(rows).set_index('ticker'), returns, drawdown


def save_chart(fig: plt.Figure, dest: Path) -> None:
    fig.tight_layout()
    fig.savefig(dest, dpi=170, bbox_inches='tight')
    plt.close(fig)


def plot_charts(prices: pd.DataFrame, returns: pd.DataFrame, metrics: pd.DataFrame,
                drawdown: pd.DataFrame, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    growth = prices.div(prices.iloc[0])
    ax = growth.plot(figsize=(12, 6), title='Growth of $1 — adjusted ETF prices')
    ax.set(ylabel='Portfolio value (USD per $1 invested)', xlabel='Date')
    save_chart(ax.figure, folder / '01_growth_of_one_dollar.png')

    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', fontsize=9,
                    color='white' if abs(corr.iloc[i,j]) > .65 else 'black')
    fig.colorbar(im, ax=ax, label='Pearson correlation')
    ax.set_title('Daily return correlations — aligned dates')
    save_chart(fig, folder / '02_correlation_heatmap.png')

    fig, ax = plt.subplots(figsize=(9, 6))
    for ticker, row in metrics.iterrows():
        ax.scatter(row['annualized_volatility'] * 100, row['cagr'] * 100, s=70)
        ax.annotate(ticker, (row['annualized_volatility'] * 100, row['cagr'] * 100), xytext=(5, 5), textcoords='offset points')
    ax.axhline(0, color='gray', linewidth=.8)
    ax.set(xlabel='Annualised volatility (%)', ylabel='CAGR (%)', title='Historical risk versus compounded return')
    save_chart(fig, folder / '03_risk_return.png')

    ax = (drawdown * 100).plot(figsize=(12, 6), title='Historical drawdowns from previous peaks')
    ax.set(xlabel='Date', ylabel='Drawdown (%)')
    save_chart(ax.figure, folder / '04_drawdowns.png')

    ax = (returns * 100).plot(kind='box', figsize=(10, 6), title='Distribution of aligned daily returns', showfliers=False)
    ax.set(ylabel='Daily return (%)', xlabel='ETF')
    save_chart(ax.figure, folder / '05_daily_return_boxplots.png')

    rolling = returns.rolling(63, min_periods=63).std() * np.sqrt(TRADING_DAYS) * 100
    ax = rolling.plot(figsize=(12, 6), title='63-observation rolling annualised volatility')
    ax.set(xlabel='Date', ylabel='Annualised volatility (%)')
    save_chart(ax.figure, folder / '06_rolling_volatility.png')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prices', type=Path, default=ROOT / 'data/processed/adjusted_close_2015_2025.parquet')
    parser.add_argument('--risk-free-rate', type=float, default=0.02, help='Annual decimal rate, e.g. 0.02')
    parser.add_argument('--min-days', type=int, default=252)
    args = parser.parse_args()
    if args.min_days < 2:
        parser.error('--min-days must be at least 2')
    prices = load_prices(args.prices)
    aligned, quality = align_prices(prices, args.min_days)
    metrics, returns, drawdown = calculate_metrics(aligned, args.risk_free_rate)
    out = ROOT / 'reports/step2'
    processed = ROOT / 'data/processed'
    out.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    aligned.to_parquet(processed / 'aligned_prices_2015_2025.parquet')
    returns.to_parquet(processed / 'aligned_daily_returns_2015_2025.parquet')
    metrics.to_csv(out / 'asset_performance_metrics.csv', float_format='%.8f')
    returns.corr().to_csv(out / 'return_correlations.csv', float_format='%.8f')
    (drawdown * 100).to_csv(out / 'drawdowns_percent.csv', float_format='%.6f', index_label='date')
    quality.update({'risk_free_rate_annual_assumed': args.risk_free_rate,
                    'risk_free_daily_compounded': (1 + args.risk_free_rate) ** (1 / TRADING_DAYS) - 1,
                    'prices_source': str(args.prices), 'all_assets_usd': True,
                    'notes': ['CAGR uses 252 observations/year, not calendar-year duration.',
                              'Daily VaR and expected shortfall are historical, not forecasts.',
                              'Common-date deletion may bridge unusually long calendar gaps; inspect gap diagnostics.',
                              'Historical prices may reflect a current ETF universe (survivorship bias).']})
    (out / 'eda_quality_report.json').write_text(json.dumps(quality, indent=2), encoding='utf-8')
    plot_charts(aligned, returns, metrics, drawdown, out / 'charts')
    print('\n=== STEP 2 DATA QUALITY ===\n' + json.dumps(quality, indent=2))
    print('\n=== ASSET METRICS (percent columns converted for display) ===')
    display = metrics[['cagr', 'annualized_volatility', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown']].copy()
    for col in ['cagr', 'annualized_volatility', 'max_drawdown']:
        display[col] *= 100
    print(display.round(3).to_string())
    print(f'\nSaved outputs to {out} and {processed}')


if __name__ == '__main__':
    main()
