"""Step 4: constrained long-only minimum variance and max Sharpe portfolios.
Weights are fitted on training returns ONLY. Run: python -m src.step4_optimization
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.step3_portfolios import ROOT, load_returns, simulate, metrics, SPLITS


def optimize(train, objective='min_vol', rf=.02, cap=.45):
    if not 0 < cap <= 1 or cap * train.shape[1] < 1: raise ValueError('Infeasible cap')
    mu=train.mean().to_numpy()*252
    cov=train.cov().to_numpy()*252
    n=len(mu)
    def vol(w): return float(np.sqrt(max(w @ cov @ w, 0)))
    def loss(w):
        v=vol(w)
        return v if objective=='min_vol' else -(float(w@mu)-rf)/max(v,1e-12)
    if objective not in ('min_vol','max_sharpe'): raise ValueError('Unknown objective')
    res=minimize(loss,np.ones(n)/n,method='SLSQP',bounds=[(0,cap)]*n,
                 constraints=[{'type':'eq','fun':lambda w: np.sum(w)-1}],
                 options={'maxiter':2000,'ftol':1e-11})
    if not res.success or not np.isclose(res.x.sum(),1,atol=1e-5):
        raise RuntimeError(f'Optimisation failed: {res.message}')
    return pd.Series(res.x,index=train.columns)


def run():
    r=load_returns(ROOT/'data/processed/aligned_daily_returns_2015_2025.parquet')
    train=r.loc['2015':'2022']
    out=ROOT/'reports/step4';out.mkdir(parents=True,exist_ok=True)
    weights={name:optimize(train,name) for name in ('min_vol','max_sharpe')}
    pd.DataFrame(weights).to_csv(out/'frozen_weights.csv')
    rows=[]
    for split,(start,end) in SPLITS.items():
        for name,w in weights.items():
            sim=simulate(r.loc[start:end],w.to_numpy(),10,'monthly')
            rows.append({'split':split,'strategy':name,**metrics(sim.net_return)})
    result=pd.DataFrame(rows);result.to_csv(out/'optimized_metrics_by_split.csv',index=False)
    (out/'methodology.json').write_text(json.dumps({'fit_period':'2015-2022 only','max_weight':0.45,'transaction_cost_bps':10,'note':'Mean estimates and covariance estimated only on train; weights frozen for later periods.'},indent=2))
    print(result.to_string(index=False));print('Saved',out)
if __name__=='__main__':run()
