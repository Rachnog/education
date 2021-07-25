import numpy as np
from scipy.stats import norm, skewnorm

def fit_gaussian(data, n = 100):
    # skew, mu, std = skewnorm.fit(data)
    mu, std = norm.fit(data)
    x = np.linspace(np.min(data), np.max(data), n)
    p = norm.pdf(x, mu, std)
    return x, p, mu, std