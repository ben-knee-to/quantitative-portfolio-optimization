import pandas as pd
import pytest
from src.step1_market_data import validate_prices


def test_valid_prices():
    prices = pd.DataFrame({'SPY': [100.0, 101.0]}, index=pd.to_datetime(['2020-01-02', '2020-01-03']))
    assert validate_prices(prices)['rows'] == 2


def test_duplicate_dates_rejected():
    prices = pd.DataFrame({'SPY': [100.0, 101.0]}, index=pd.to_datetime(['2020-01-02', '2020-01-02']))
    with pytest.raises(ValueError, match='Duplicate'):
        validate_prices(prices)


def test_nonpositive_prices_rejected():
    prices = pd.DataFrame({'SPY': [100.0, 0.0]}, index=pd.to_datetime(['2020-01-02', '2020-01-03']))
    with pytest.raises(ValueError, match='Non-positive'):
        validate_prices(prices)
