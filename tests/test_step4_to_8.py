import numpy as np
import pandas as pd
import pytest
from src.step4_optimization import optimize
from src.step5_risk import risk_stats, bootstrap_terminal
from src.step6_strategies import momentum_backtest
from src.step7_machine_learning import features

def sample():
    rng=np.random.default_rng(7)
    return pd.DataFrame(rng.normal(.0002,.01,(300,4)),index=pd.bdate_range('2020-01-01',periods=300),columns=list('ABCD'))

def test_optimizer_long_only_and_capped():
    w=optimize(sample(),cap=.5)
    assert np.isclose(w.sum(),1) and w.min()>=-1e-7 and w.max()<=.500001

def test_risk_outputs():
    x=sample()['A']
    assert risk_stats(x)['daily_var_95_loss']>=0
    assert 0<=bootstrap_terminal(x,paths=50,days=10)['probability_loss']<=1

def test_momentum_no_future_leakage():
    r=sample();a=momentum_backtest(r,lookback=20)
    modified=r.copy();modified.iloc[-1,0]=.5
    b=momentum_backtest(modified,lookback=20)
    pd.testing.assert_frame_equal(a.iloc[:-1],b.iloc[:-1])

def test_ml_features_lagged():
    r=sample().rename(columns={'A':'SPY','B':'QQQ'})
    x,y=features(r)
    assert np.isnan(x.lag1.iloc[0])
    assert np.isclose(x.lag1.iloc[2],r.SPY.iloc[1])
