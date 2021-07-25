import pandas as pd
import numpy as np

from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import objective_functions

import cvxpy as cp
from portfoliolab.clustering.hrp import HierarchicalRiskParity
from sklearn.decomposition import PCA

import tensorflow as tf
import tensorflow.keras as tfk

class CapitalAllocatorEqual:

    def __init__(self):
        pass

    def fit(self, window_fit):
        self.window_fit = window_fit
        self.n_assets_window = window_fit.shape[-1]
    
    def get_weights(self):
        optimal_weights = np.ones(self.n_assets_window) / self.n_assets_window
        return pd.DataFrame({
              'ticker': self.window_fit.columns,
              'weight': optimal_weights
        })

class CapitalAllocatorGMV:

    def __init__(self, shrinkage = False):
        self.shrinkage = shrinkage

    def fit(self, window_fit):
        self.window_fit = window_fit
        self.mu = expected_returns.mean_historical_return(self.window_fit, returns_data=True)
        self.S = risk_models.sample_cov(self.window_fit, returns_data=True)
        if self.shrinkage:
            self.S = CovarianceShrinkage(self.window_fit, returns_data=True).ledoit_wolf()
    
    def get_weights(self):
        # Optimize for maximal Sharpe ratio
        ef = EfficientFrontier(self.mu, self.S)
        optimal_weights = ef.min_volatility()
        optimal_weights = pd.DataFrame([dict(optimal_weights)]).T.values.flatten()
        return pd.DataFrame({
              'ticker': self.window_fit.columns,
              'weight': optimal_weights
        })

class CapitalAllocatorEF:

    def __init__(self, shrinkage = False):
        self.shrinkage = shrinkage

    def fit(self, window_fit):
        self.window_fit = window_fit
        self.mu = expected_returns.mean_historical_return(self.window_fit, returns_data=True)
        self.S = risk_models.sample_cov(self.window_fit, returns_data=True)
        if self.shrinkage:
            self.S = CovarianceShrinkage(self.window_fit, returns_data=True).ledoit_wolf()
    
    def get_weights(self):
        # Optimize for maximal Sharpe ratio
        ef = EfficientFrontier(self.mu, self.S)
        optimal_weights = ef.max_sharpe()
        optimal_weights = pd.DataFrame([dict(optimal_weights)]).T.values.flatten()
        return pd.DataFrame({
              'ticker': self.window_fit.columns,
              'weight': optimal_weights
        })

class CapitalAllocatorCustom:

    def __init__(self, shrinkage = False, shorting = False):
        self.shrinkage = shrinkage
        self.shorting = shorting

    def fit(self, window_fit):
        self.window_fit = window_fit
        self.mu = expected_returns.mean_historical_return(self.window_fit, returns_data=True)
        self.S = risk_models.sample_cov(self.window_fit, returns_data=True)
        if self.shrinkage:
            self.S = CovarianceShrinkage(self.window_fit, returns_data=True).ledoit_wolf()
    
    def get_weights(self):

        def decorrelate(weights, corr_matrix):
            return np.sqrt(np.dot(weights.T, np.dot(corr_matrix, weights)))

        def deviation_risk_parity(w, cov_matrix):
            diff = w * np.dot(cov_matrix, w) - (w * np.dot(cov_matrix, w)).reshape(-1, 1)
            return (diff ** 2).sum().sum()

        # Optimize for maximal Sharpe ratio
        if self.shorting:
            ef = EfficientFrontier(self.mu, self.S, weight_bounds=(-1, 1))
        else:
            ef = EfficientFrontier(self.mu, self.S)

        ef.add_objective(objective_functions.L2_reg, gamma=10)
        # ef.nonconvex_objective(decorrelate, self.window_fit.corr())
        ef.nonconvex_objective(deviation_risk_parity, self.S)
        optimal_weights = ef.clean_weights()

        optimal_weights = pd.DataFrame([dict(optimal_weights)]).T.values.flatten()
        return pd.DataFrame({
              'ticker': self.window_fit.columns,
              'weight': optimal_weights
        })

class CapitalAllocatorHRP:

    def __init__(self, shrinkage = True):
        self.shrinkage = shrinkage

    def fit(self, window_fit):
        self.window_fit = window_fit
        self.S = risk_models.sample_cov(self.window_fit, returns_data=True)
        if self.shrinkage:
            self.S = CovarianceShrinkage(self.window_fit, returns_data=True).ledoit_wolf()
    
    def get_weights(self):

        self.hrp = HierarchicalRiskParity()
        self.hrp.allocate(covariance_matrix=self.S, linkage='ward')
        optimal_weights = self.hrp.weights[self.window_fit.columns].T.values.flatten()

        return pd.DataFrame({
              'ticker': self.window_fit.columns,
              'weight': optimal_weights
        })

class CapitalAllocatorPCA:

    def __init__(self, shrinkage = True, C = 10):
        self.shrinkage = shrinkage
        self.C = C

    def fit(self, window_fit):
        self.window_fit = window_fit
        self.S = risk_models.sample_cov(self.window_fit, returns_data=True)
        if self.shrinkage:
            self.S = CovarianceShrinkage(self.window_fit, returns_data=True).ledoit_wolf()
    
    def get_weights(self, compontent):

        self.pca = PCA(self.C)
        returns_train_pca = self.pca.fit_transform(self.S)
        pcs = self.pca.components_
        w = pcs[compontent, :]
        optimal_weights = w / sum(np.abs(w))

        return pd.DataFrame({
              'ticker': self.window_fit.columns,
              'weight': optimal_weights
        })


class CapitalAllocatorAE:

    def __init__(self, C = 100):
        self.C = C

    def fit(self, window_fit):
        self.window_fit = window_fit
    
    def get_weights(self):
        # connect all layers
        input_img = tfk.Input(shape=(self.window_fit.shape[1], ))
        encoded = tfk.layers.Dense(self.C * 2, activation='relu')(input_img)
        encoded = tfk.layers.Dense(self.C, activation='relu', kernel_regularizer=tfk.regularizers.l2(1e-3))(encoded)
        decoded = tfk.layers.Dense(self.C * 2, activation= 'relu')(encoded)
        decoded = tfk.layers.Dense(self.window_fit.shape[1], activation = 'linear')(decoded)
        # construct and compile AE model
        self.autoencoder = tfk.Model(input_img, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')

        history = self.autoencoder.fit(self.window_fit, self.window_fit, epochs=100, batch_size=64, verbose=False)
        reconstruct = self.autoencoder.predict(self.window_fit)
  
        communal_information = []
        for i in range(0, len(self.window_fit.columns)):
            diff = np.linalg.norm((self.window_fit.iloc[:,i] - reconstruct[:,i])) # 2 norm difference
            communal_information.append(float(diff))
        communal_information = np.array(communal_information)

        weights_ae = (1 / communal_information)
        weights_ae = np.array(weights_ae) / sum(np.abs(weights_ae))

        return pd.DataFrame({
              'ticker': self.window_fit.columns,
              'weight': weights_ae
        })