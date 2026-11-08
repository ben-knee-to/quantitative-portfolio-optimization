"""Step 8: assemble cross-step results without changing strategies or models."""
import pandas as pd
from src.step3_portfolios import ROOT

def run():
    sources={'benchmarks':'step3/portfolio_metrics_by_split.csv','optimized':'step4/optimized_metrics_by_split.csv','momentum':'step6/momentum_metrics.csv'}
    rows=[]
    for group,relative in sources.items():
        path=ROOT/'reports'/relative
        if not path.exists():raise FileNotFoundError(f'Run the corresponding step first: {path}')
        frame=pd.read_csv(path)
        frame['group']=group
        if 'portfolio' not in frame:frame['portfolio']=frame.get('strategy','Momentum')
        rows.append(frame[['group','split','portfolio','cagr','annualized_volatility','sharpe_ratio','sortino_ratio','max_drawdown']])
    out=ROOT/'reports/step8';out.mkdir(parents=True,exist_ok=True)
    df=pd.concat(rows,ignore_index=True);df.to_csv(out/'final_comparison.csv',index=False)
    print(df[df.split=='test'].round(4).to_string(index=False))
    print('CAUTION: 2025 asset EDA was previously inspected; do not describe test as pristine.')
if __name__=='__main__':run()
