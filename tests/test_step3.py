import numpy as np
import pandas as pd
import pytest
from src.step3_portfolios import simulate, target_weights, metrics, WEIGHTS, EXPECTED


def sample_returns():
    return pd.DataFrame({'A':[.10,0.,.05], 'B':[0.,.02,-.02]},
                        index=pd.to_datetime(['2022-01-28','2022-01-31','2022-02-01']))


def test_allocations_sum_to_one():
    for a in WEIGHTS.values():
        w = target_weights(a,pd.Index(EXPECTED))
        assert np.isclose(w.sum(),1) and (w>=0).all()


def test_rebalance_at_first_trading_date_of_month():
    sim = simulate(sample_returns(),np.array([.5,.5]),0)
    assert sim.rebalanced.tolist() == [True,False,True]
    assert np.isclose(sim.turnover.iloc[0],1)
    assert sim.turnover.iloc[1] == 0
    assert sim.turnover.iloc[2] > 0


def test_no_lookahead_first_return():
    sim = simulate(sample_returns(),np.array([.5,.5]),0)
    assert sim.net_return.iloc[0] == pytest.approx(.05)


def test_transaction_cost_reduces_wealth():
    r = sample_returns()
    w = np.array([.5,.5])
    net = simulate(r,w,10)
    gross = simulate(r,w,0)
    assert net.net_wealth.iloc[-1] < gross.net_wealth.iloc[-1]
    assert net.net_return.iloc[0] == pytest.approx((1-.001)*(1+.05)-1)


def test_buy_hold_no_extra_rebalance():
    sim = simulate(sample_returns(),np.array([1.,0.]),0,rebalance='none')
    assert sim.rebalanced.sum() == 1
    assert sim.net_wealth.iloc[-1] == pytest.approx(10000*1.1*1.05)


def test_flat_return_zero_volatility_and_drawdown():
    r = pd.Series([0.]*30)
    result = metrics(r)
    assert result['annualized_volatility'] == 0
    assert result['max_drawdown'] == 0
    assert np.isnan(result['sharpe_ratio'])


def test_drawdown_includes_starting_wealth():
    result = metrics(pd.Series([-.2,.1]))
    assert result['max_drawdown'] == pytest.approx(-.2)


def test_invalid_weights_rejected():
    with pytest.raises(ValueError):
        target_weights({'SPY':1.2},pd.Index(EXPECTED))


def test_cost_rejected():
    with pytest.raises(ValueError):
        simulate(sample_returns(),np.array([.5,.5]),-1)
