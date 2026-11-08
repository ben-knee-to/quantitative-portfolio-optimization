"""Run: streamlit run app/streamlit_app.py"""
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
ROOT=Path(__file__).resolve().parents[1]
st.set_page_config(page_title='Quant Portfolio Analytics',layout='wide')
st.title('Quantitative Portfolio Analytics')
st.caption('Historical educational research, not investment advice. 2025 is not a pristine unseen holdout.')
section=st.sidebar.selectbox('Report',['Benchmarks','Optimisation','Risk','Momentum','Machine learning','Final comparison'])
files={'Benchmarks':'step3/portfolio_metrics_by_split.csv','Optimisation':'step4/optimized_metrics_by_split.csv',
       'Risk':'step5/training_risk_scenarios.csv','Momentum':'step6/momentum_metrics.csv',
       'Machine learning':'step7/ml_metrics.csv','Final comparison':'step8/final_comparison.csv'}
path=ROOT/'reports'/files[section]
if not path.exists():st.warning(f'Missing {path.relative_to(ROOT)}. Run the corresponding script first.');st.stop()
df=pd.read_csv(path);st.dataframe(df,use_container_width=True)
if 'split' in df and 'sharpe_ratio' in df:
    group='portfolio' if 'portfolio' in df else ('strategy' if 'strategy' in df else 'split')
    st.plotly_chart(px.bar(df,x=group,y='sharpe_ratio',color='split',barmode='group',title='Sharpe ratio comparison'),use_container_width=True)
if section=='Benchmarks':
    daily=ROOT/'reports/step3/portfolio_daily_paths.csv'
    if daily.exists():
        paths=pd.read_csv(daily,parse_dates=['date']);split=st.selectbox('Period',paths.split.unique())
        s=paths[paths.split==split]
        st.plotly_chart(px.line(s,x='date',y='net_wealth',color='portfolio',title='Net portfolio wealth'),use_container_width=True)
