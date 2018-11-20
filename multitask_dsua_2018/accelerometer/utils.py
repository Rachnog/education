from feature_calculators import standard_deviation, absolute_sum_of_changes

import numpy as np
import os

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

import random

random.seed(42)
np.random.seed(42)

data_folder = 'HMP_Dataset'

def load_raw_data(data_folder):
    '''
        Loading raw accelerometer data from HMP dataset
    '''
    X_raw, Y_raw = [], []
    for i, (root, dirs, files) in enumerate(os.walk(data_folder)):
        if i == 0:
            continue
        path = root.split(os.sep)
        data_class = '.'.join(path)
        for file in files:
            full_path = root + '/' + file
            data = open(full_path).readlines()
            X_raw.append(data)
            Y_raw.append(data_class)
    return X_raw, Y_raw


def prepare_x_y(X_raw, Y_raw):
    '''
        Preparing pairs / triples / .. of inputs and outputs for a ML model
    '''
    X, Y, Y_std, Y_soc, Y_fourier = [], [], [], [], []
    
    for x_raw in X_raw:
        x_buf = np.array([[int(xi) for xi in x.strip().split()] for x in x_raw])
        X.append(x_buf)
        Y_std.append(standard_deviation(x_buf[:, 0])) # standard deviation of X axis 
        Y_soc.append(absolute_sum_of_changes(x_buf[:, 0])) # sum of changes of X axis
        Y_fourier.append(np.abs(np.fft.rfft(x_buf[:, 0]))[:50]) # Fourier coef (5) of X axis
    
    le = LabelEncoder()
    Y = le.fit_transform(Y_raw)
    Y = to_categorical(Y)
    Y_std, Y_soc, Y_fourier = np.array(Y_std), np.array(Y_soc), np.array(Y_fourier)
    return X, Y, Y_std, Y_soc, Y_fourier


def prepare_train_test(X, Y, Y_std, Y_soc, Y_fourier, random_state = 42):
    '''
        Padding sequences and splitting into train / test sets
    '''
    X = pad_sequences(X, value = [0, 0, 0])
    X_train, X_test, Y_train, Y_test, Y_train_std, Y_test_std, \
                     Y_train_soc, Y_test_soc, Y_train_fourier, Y_test_fourier = train_test_split(X, 
                                                                           Y, Y_std, Y_soc, Y_fourier,
                                                                           random_state=random_state)
        
    fourier_scaler = MinMaxScaler()
    Y_train_fourier = fourier_scaler.fit_transform(Y_train_fourier)
    Y_test_fourier = fourier_scaler.transform(Y_test_fourier)
        
    X_train, X_test, Y_train, Y_test, Y_train_std, Y_test_std, \
                     Y_train_soc, Y_test_soc, Y_train_fourier, Y_test_fourier = np.array(X_train), \
                                                                                np.array(X_test), \
                                                                                np.array(Y_train), \
                                                                                np.array(Y_test), \
                                                                                np.array(Y_train_std), \
                                                                                np.array(Y_test_std), \
                                                                                np.array(Y_train_soc), \
                                                                                np.array(Y_test_soc), \
                                                                                np.array(Y_train_fourier), \
                                                                                np.array(Y_test_fourier)
    return X_train, X_test, Y_train, Y_test, Y_train_std, Y_test_std, \
                     Y_train_soc, Y_test_soc, Y_train_fourier, Y_test_fourier
    
    
def test_feature(Y, Y2, fourier = False):
    from sklearn.linear_model import LogisticRegression
    
    lr = LogisticRegression()
    if not fourier:
        lr.fit(Y2.reshape(-1, 1), [np.argmax(y) for y in Y]) # np.argmax() for sklearn's LR
        pred = lr.predict(Y2.reshape(-1, 1))
    else:
        lr.fit(Y2, [np.argmax(y) for y in Y])
        pred = lr.predict(Y2)

    print classification_report([np.argmax(y) for y in Y], pred)