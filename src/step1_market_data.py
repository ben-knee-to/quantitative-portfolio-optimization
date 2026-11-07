"""Step 1: acquire, validate and profile historical market prices.
Run: python -m src.step1_market_data
"""
from pathlib import Path
import json
import logging
import time
import pandas as pd
import numpy as np
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'
REPORTS = ROOT / 'reports'
START, END = '2015-01-01', '2026-01-01'  # end is exclusive; reserve 2026 for later tests
ASSETS = {
    'SPY': ('US equity', 'USD'),
    'QQQ': ('US technology equity', 'USD'),
    'IWM': ('US small-cap equity', 'USD'),
    'EFA': ('Developed ex-US equity', 'USD'),
    'EEM': ('Emerging markets equity', 'USD'),
    'AGG': ('US bonds', 'USD'),
    'GLD': ('Gold', 'USD'),
    'VNQ': ('US REIT', 'USD'),
}
# All portfolio assets are USD denominated. JSE/ZAR assets will be a separate extension.
LOG = logging.getLogger('market_data')


def validate_prices(df: pd.DataFrame) -> dict:
    if df.empty:
        raise ValueError('No market prices downloaded')
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError('Expected DatetimeIndex')
    if df.index.has_duplicates:
        raise ValueError('Duplicate dates detected')
    if not df.index.is_monotonic_increasing:
        raise ValueError('Dates are not sorted')
    if (df.dropna() <= 0).any().any():
        raise ValueError('Non-positive prices detected')
    return {
        'rows': int(len(df)),
        'first_date': df.index.min().date().isoformat(),
        'last_date': df.index.max().date().isoformat(),
        'missing_by_ticker': {k: int(v) for k, v in df.isna().sum().items()},
        'available_by_ticker': {k: int(v) for k, v in df.notna().sum().items()},
    }


def fetch_one(ticker: str, retries: int = 3) -> pd.DataFrame:
    for attempt in range(1, retries + 1):
        try:
            frame = yf.download(
                ticker, start=START, end=END, interval='1d',
                auto_adjust=True, actions=False, progress=False,
                threads=False, timeout=25,
            )
            if frame is None or frame.empty:
                raise ValueError('Empty download')
            if isinstance(frame.columns, pd.MultiIndex):
                # yfinance default can return (Price, Ticker) or (Ticker, Price).
                matches = [c for c in frame.columns if 'Close' in c]
                if len(matches) != 1:
                    raise ValueError(f'Unexpected Close columns: {matches}')
                close = frame[matches[0]]
            else:
                close = frame['Close']
            close = pd.to_numeric(close, errors='coerce')
            close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
            close = close.sort_index()
            close = close[~close.index.duplicated(keep='last')]
            result = close.rename(ticker).to_frame()
            result.index.name = 'date'
            return result
        except Exception as exc:
            LOG.warning('%s attempt %d/%d failed: %s', ticker, attempt, retries, exc)
            if attempt < retries:
                time.sleep(attempt * 2)
    raise RuntimeError(f'Could not download {ticker} after {retries} attempts')


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    for directory in (RAW, PROCESSED, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    series = []
    failed = {}
    for ticker in ASSETS:
        LOG.info('Downloading %s', ticker)
        try:
            series.append(fetch_one(ticker))
        except RuntimeError as exc:
            failed[ticker] = str(exc)
    if failed:
        (REPORTS / 'download_errors.json').write_text(json.dumps(failed, indent=2))
        raise RuntimeError(f'Download incomplete: {failed}. Retry later; no partial dataset saved.')
    prices = pd.concat(series, axis=1).sort_index()
    prices = prices.loc[(prices.index >= START) & (prices.index < END)]
    quality = validate_prices(prices)
    quality['currency'] = 'USD'
    quality['price_definition'] = 'yfinance auto_adjust=True adjusted Close'
    quality['requested_start'] = START
    quality['requested_end_exclusive'] = END
    quality['tickers'] = list(ASSETS)
    quality['missing_fraction_by_ticker'] = {k: round(v / len(prices), 5) for k, v in quality['missing_by_ticker'].items()}
    # Preserve gaps: no forward fill across exchange holidays or missing observations.
    prices.to_csv(RAW / 'adjusted_close_2015_2025.csv', index_label='date')
    prices.to_parquet(PROCESSED / 'adjusted_close_2015_2025.parquet')
    # pct_change(fill_method=None) avoids fabricating returns across missing data.
    returns = prices.pct_change(fill_method=None)
    returns.to_csv(PROCESSED / 'daily_returns_unaligned.csv', index_label='date')
    quality['complete_case_days'] = int(prices.dropna().shape[0])
    quality['duplicate_dates'] = int(prices.index.duplicated().sum())
    (REPORTS / 'data_quality.json').write_text(json.dumps(quality, indent=2))
    summary = pd.DataFrame({
        'observations': prices.count(),
        'missing_days': prices.isna().sum(),
        'first_valid': prices.apply(lambda s: s.first_valid_index().date().isoformat() if s.first_valid_index() is not None else None),
        'last_valid': prices.apply(lambda s: s.last_valid_index().date().isoformat() if s.last_valid_index() is not None else None),
        'min_adjusted_close': prices.min(),
        'max_adjusted_close': prices.max(),
    })
    summary.index.name = 'ticker'
    summary.to_csv(REPORTS / 'asset_profile.csv')
    print('\n=== STEP 1 DATA QUALITY ===')
    print(json.dumps(quality, indent=2))
    print('\n=== ASSET PROFILE ===')
    print(summary.to_string())
    print('\nFiles saved under data/raw, data/processed, and reports.')


if __name__ == '__main__':
    main()
