import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import scipy.stats as ss

class SingleInstrumentWFScorer:

    def __init__(self, periods_per_year = 252):
        self.periods_per_year = periods_per_year

    def returns(self, r):
        return r.sum()

    def annualized_returns(self, r):
        compounded_growth = (1+r).prod()
        n_periods = r.shape[0]
        return compounded_growth**(self.periods_per_year/n_periods)-1

    def annualized_volatility(self, r):
        return r.std() * np.sqrt(self.periods_per_year)

    def drawdown(self, r):
        wealth_index = 1000*(1+r).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks)/previous_peaks
        return pd.DataFrame({"Wealth": wealth_index, 
                            "Previous Peak": previous_peaks, 
                            "Drawdown": drawdowns})

    def semideviation(self, r):
        return r[r < 0].std(ddof=0)

    def sharpe(self, r):
        return self.annualized_returns(r) / self.annualized_volatility(r)

    def risk_report(self, r):
        return pd.DataFrame([{
            'total_return': self.returns(r),
            'annualized_return': self.annualized_returns(r),
            'annualized_volatility': self.annualized_volatility(r),
            'semideviation': self.semideviation(r),
            'max_drawdown': self.drawdown(r)['Drawdown'].min(),
            'annualized_sharpe': self.sharpe(r)
        }])

class CompareInstrumentsWF:

    def __init__(self, target_sharpe = 1.0, periods_per_year = 252):
        self.periods_per_year = periods_per_year
        self.target_sharpe = target_sharpe

    def information_ratio(self, returns, benchmark):
        diff = returns - benchmark
        return diff.mean() / diff.std()

    def psr(self, returns, benchmark):
        returns = returns.dropna()
        benchmark = benchmark.dropna()
        N = len(returns)
        sharpe = returns.mean() / returns.std() * np.sqrt(self.periods_per_year)
        returns_skew = skew(returns)
        returns_kurt = kurtosis(returns)
        value = (
            (sharpe - self.target_sharpe)
            * np.sqrt(N - 1)
            / np.sqrt(1.0 - returns_skew * sharpe + sharpe ** 2 * (returns_kurt - 1) / 4.0)
        )
        psr = ss.norm.cdf(value, 0, 1)
        return psr

    def risk_report(self, r, b):
        return pd.DataFrame([{
            'info_ratio': self.information_ratio(r, b),
            'proba_sharpe': self.psr(r, b)
        }])

