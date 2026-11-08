"""Step 6: causal monthly momentum rotation; parameters are fixed, no future data used."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.step3_portfolios import ROOT, load_returns, metrics, SPLITS


def momentum_backtest(returns, lookback=126, cost_bps=10):
    if lookback<20 or cost_bps<0:raise ValueError('Invalid configuration')
    # Signal is formed at previous close using previous lookback completed returns.
    scores=(1+returns).rolling(lookback).apply(np.prod,raw=True).shift(1)-1
    rows=[];current=None;wealth=1.;last_month=None
    for dt,day in returns.iterrows():
        month=dt.to_period('M');turnover=0.
        if month!=last_month:
            available=scores.loc[dt].dropna()
            new=available.idxmax() if len(available)==len(returns.columns) else None
            turnover=0 if new==current else (1 if current is None or new is None else 2)
            current=new
        gross=float(day[current]) if current is not None else 0.
        net=(1-cost_bps/10000*turnover)*(1+gross)-1
        wealth*=1+net
        rows.append({'date':dt,'selected_asset':current or 'CASH','net_return':net,'gross_return':gross,'turnover':turnover,'wealth':wealth})
        last_month=month
    return pd.DataFrame(rows).set_index('date')


def run():
    r=load_returns(ROOT/'data/processed/aligned_daily_returns_2015_2025.parquet')
    # Warmup history is carried into later periods, with signals shifted by one day.
    sim=momentum_backtest(r);out=ROOT/'reports/step6';out.mkdir(parents=True,exist_ok=True)
    sim.to_csv(out/'momentum_daily.csv')
    rows=[]
    for split,(start,end) in SPLITS.items():
        s=sim.loc[start:end,'net_return']
        rows.append({'split':split,**metrics(s)})
    pd.DataFrame(rows).to_csv(out/'momentum_metrics.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,5));sim.wealth.plot(ax=ax,title='Causal momentum rotation (illustrative)');fig.tight_layout();fig.savefig(out/'momentum_wealth.png');plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':run()
