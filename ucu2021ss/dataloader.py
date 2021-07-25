import yfinance as yf
import pandas as pd
import numpy as np

class YFinanceDataLoader:

    def __init__(self, assets_list, start_date = '2000-01-01', end_date = '2021-08-01', column = 'Adj Close'):
        self.assets_list = assets_list
        self.start_date = start_date
        self.end_date = end_date
        self.column = column

    def load_all(self):
        all_dfs = {}
        for etf in self.assets_list:
            try:
                df = yf.download(etf.split('.')[0], 
                                  start=self.start_date, 
                                  end=self.end_date)
                df_close = df[self.column].to_frame()
                df_close.columns = [etf]
                all_dfs[etf] = df_close
            except Exception as e:
                print(etf, e)
        return pd.concat(list(all_dfs.values()), axis=1)

    def load_single(self, asset):
        return yf.download(asset, start=self.start_date, end=self.end_date)[self.column]