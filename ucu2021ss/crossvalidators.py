from sklearn.model_selection import BaseCrossValidator, GroupKFold
from itertools import combinations

import pandas as pd
import numpy as np

class PurgeCVByEras(BaseCrossValidator):

    def __init__(self, eras, n_eras_train, n_eras_test, split_embargo, tr_te_embargo):
        self.eras = eras
        self.n_eras = len(self.eras.unique())
        self.n_eras_train = n_eras_train
        self.n_eras_test = n_eras_test
        self.split_embargo = split_embargo
        self.tr_te_embargo = tr_te_embargo

    def get_indices_of_era(self, era):
        return self.eras.index[self.eras == era].tolist()

    def split(self, X, y=None, groups=None):
        for i in range(0, self.n_eras, (self.n_eras_train + self.n_eras_test + self.split_embargo+self.tr_te_embargo)):

            train_eras = [i+n for n in range(self.n_eras_train)]
            test_eras = [i+m+self.n_eras_train+self.tr_te_embargo for m in range(self.n_eras_test)]
            train_indices = []

            test_limit = 0
            if max(train_eras) >= self.n_eras:
                break
            elif max(test_eras) >= self.n_eras:
                max_index = test_eras.index(max(test_eras))
                test_eras = test_eras[:max_index]

            for e in train_eras:
                e_indices = self.get_indices_of_era(e)
                train_indices.extend(e_indices)

            test_indices = []
            for e in test_eras:
                e_indices = self.get_indices_of_era(e)
                test_indices.extend(e_indices)

            if len(train_indices) != 0 and len(test_indices) != 0:
                yield(train_indices, test_indices) 
            
    def get_n_splits(self):
        return len(range(0, self.n_eras, (self.n_eras_train + self.n_eras_test + self.split_embargo + self.tr_te_embargo)))


class CPCV(BaseCrossValidator):

    def __init__(self, eras, N, k):
        self.eras = eras
        self.N = N
        self.k = k

    def split(self, X=None, y=None, groups=None):
        len_diff = abs(len(X) - len(self.eras))
        comb = list(combinations(range(self.N), self.N-self.k))
        all_splits = range(self.N)

        for combination in comb:
            train_indices, test_indices = [], []
            for c in combination:
                indices_train = list(np.where(self.eras == c)[0])
                train_indices.extend(indices_train)
            for t in list(set(all_splits) - set(combination)):
                indices_test = list(np.where(self.eras == t)[0])
                test_indices.extend(indices_test)

            if len(train_indices) != 0 and len(test_indices) != 0:
                yield(train_indices, test_indices) 
              
    def get_n_splits(self):
        comb = combinations(range(self.N), self.N-self.k)
        return len(list(comb))