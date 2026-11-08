"""Step 5: historical VaR/ES and reproducible bootstrap risk scenarios."""
import numpy as np
import pandas as pd
from src.step3_portfolios import ROOT, load_returns, WEIGHTS, target_weights


def risk_stats(returns, alpha=.05):
    x=np.asarray(returns,dtype=float)
    if x.ndim!=1 or len(x)<30 or not np.isfinite(x).all() or not 0<alpha<1: raise ValueError('Invalid risk input')
    threshold=float(np.quantile(x,alpha))
    return {'daily_var_95_loss':max(0,-threshold),'daily_expected_shortfall_95_loss':max(0,-float(x[x<=threshold].mean())),'daily_volatility':float(x.std(ddof=1))}


def bootstrap_terminal(returns, days=252, paths=3000, seed=42):
    rng=np.random.default_rng(seed)
    x=np.asarray(returns,dtype=float)
    if len(x)<30 or (x<=-1).any() or not np.isfinite(x).all(): raise ValueError('Invalid returns')
    final=np.ones(paths)
    for _ in range(days): final*=1+rng.choice(x,size=paths,replace=True)
    return {'p05':float(np.quantile(final,.05)),'median':float(np.median(final)),'p95':float(np.quantile(final,.95)),'probability_loss':float(np.mean(final<1))}


def run():
    r=load_returns(ROOT/'data/processed/aligned_daily_returns_2015_2025.parquet')
    train=r.loc['2015':'2022'];rows=[]
    for name,allocation in WEIGHTS.items():
        w=target_weights(allocation,train.columns)
        x=train.to_numpy()@w
        rows.append({'portfolio':name,**risk_stats(x),**{f'bootstrap_{k}':v for k,v in bootstrap_terminal(x).items()}})
    out=ROOT/'reports/step5';out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows);df.to_csv(out/'training_risk_scenarios.csv',index=False)
    print(df.to_string(index=False));print('Saved',out)
if __name__=='__main__':run()
