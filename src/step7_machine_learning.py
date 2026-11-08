"""Step 7: illustrative next-day direction prediction, chronological training only."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from src.step3_portfolios import ROOT, load_returns


def features(r):
    s=r['SPY']
    x=pd.DataFrame({'lag1':s.shift(1),'lag2':s.shift(2),'momentum5':s.shift(1).rolling(5).mean(),
                    'vol20':s.shift(1).rolling(20).std(), 'qqq_lag1':r['QQQ'].shift(1)},index=r.index)
    y=(s>0).astype(int)
    # Current day's return is target; all features come from prior sessions.
    return x,y


def run():
    r=load_returns(ROOT/'data/processed/aligned_daily_returns_2015_2025.parquet')
    x,y=features(r);valid=x.notna().all(axis=1);x=x[valid];y=y[valid]
    masks={'train':x.index.year<=2022,'validation':(x.index.year>=2023)&(x.index.year<=2024),'test':x.index.year==2025}
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000))
    model.fit(x.loc[masks['train']],y.loc[masks['train']])
    rows=[];pred=[]
    for split,mask in masks.items():
        prob=model.predict_proba(x.loc[mask])[:,1];actual=y.loc[mask]
        rows.append({'split':split,'n':len(actual),'accuracy':accuracy_score(actual,prob>=.5),
                     'auc':roc_auc_score(actual,prob) if actual.nunique()>1 else np.nan,
                     'brier':brier_score_loss(actual,prob)})
        pred.append(pd.DataFrame({'date':actual.index,'split':split,'actual':actual.to_numpy(),'probability_up':prob}))
    out=ROOT/'reports/step7';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'ml_metrics.csv',index=False)
    pd.concat(pred).to_csv(out/'ml_predictions.csv',index=False)
    (out/'methodology.json').write_text(json.dumps({'target':'SPY next session positive return','features':'lagged daily returns, trailing mean and volatility','model':'StandardScaler + LogisticRegression fitted 2015-2022 only','warning':'Predictive accuracy is not trading profitability; no parameter tuning on test.'},indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':run()
