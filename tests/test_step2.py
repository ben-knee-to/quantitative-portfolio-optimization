import numpy as np
import pandas as pd
import pytest
from src.step2_eda import EXPECTED, align_prices, calculate_metrics, load_prices


def sample_prices(n=300):
    dates = pd.bdate_range('2020-01-01', periods=n, name='date')
    values = {t: 100 * (1 + 0.0002 * (i + 1)) ** np.arange(n) for i, t in enumerate(EXPECTED)}
    return pd.DataFrame(values, index=dates)


def test_flat_prices_have_zero_volatility_and_drawdown():
    prices = sample_prices()
    prices['SPY'] = 100.0
    metrics, returns, dd = calculate_metrics(prices)
    assert metrics.loc['SPY', 'annualized_volatility'] == pytest.approx(0)
    assert metrics.loc['SPY', 'max_drawdown'] == pytest.approx(0)
    assert np.isnan(metrics.loc['SPY', 'sharpe_ratio'])
    assert returns.shape == (299, 8)
    assert dd.shape == (300, 8)


def test_constant_growth_cagr():
    metrics, _, _ = calculate_metrics(sample_prices(), annual_rf=0)
    assert metrics.loc['SPY', 'cagr'] == pytest.approx((1.0002) ** 252 - 1)
    assert metrics.loc['SPY', 'total_return'] == pytest.approx((1.0002) ** 299 - 1)


def test_alignment_removes_incomplete_dates():
    prices = sample_prices()
    prices.iloc[4, 1] = np.nan
    aligned, quality = align_prices(prices)
    assert len(aligned) == 299
    assert quality['dropped_price_dates'] == 1
    assert aligned.notna().all().all()


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_prices(tmp_path / 'absent.parquet')


def test_negative_prices_rejected(tmp_path):
    prices = sample_prices()
    prices.iloc[0, 0] = -1
    path = tmp_path / 'bad.csv'
    prices.to_csv(path)
    with pytest.raises(ValueError, match='Nonpositive'):
        load_prices(path)
