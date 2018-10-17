import numpy as np
import glob

from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Permute
from keras.layers import Merge, Input, concatenate
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Convolution1D, MaxPooling1D, AtrousConvolution1D, RepeatVector
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *
from keras.constraints import *
from keras import regularizers
from keras import losses


from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
from keras import backend as K


import matplotlib.pylab as plt
from sklearn.metrics import classification_report
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split



### BEFORE YOU HAVE TO LOAD DATA
### In current example X_clean is array with "good" time series we gonna train autoencoder on,
### and X_bad_filtered are bad examples that have to be classified as anomalies



### AUTOENCODER ARCHITECTURE ###

input_img = Input(shape=(len(X_train[0]), 1))

x = Conv1D(16, 6, activation='relu', padding='same')(input_img)
x = MaxPooling1D(2, padding='same')(x)
x = Conv1D(16, 6, activation='relu', padding='same')(x)
x = MaxPooling1D(2, padding='same')(x)
x = Conv1D(16, 6, activation='relu', padding='same')(x)
x = MaxPooling1D(2, padding='same')(x)
x = Conv1D(32, 6, activation='relu', padding='same')(x)
encoded = MaxPooling1D(5, padding='same')(x)


x = Conv1D(32, 6, activation='relu', padding='same')(encoded)
x = UpSampling1D(5)(x)
x = Conv1D(16, 6, activation='relu', padding='same')(x)
x = UpSampling1D(2)(x)
x = Conv1D(16, 6, activation='relu', padding='same')(x)
x = UpSampling1D(2)(x)
x = Conv1D(16, 6, activation='relu', padding='same')(x)
x = UpSampling1D(2)(x)
decoded = Conv1D(1, 6, activation='linear', padding='same')(x)

opt = Nadam(0.002)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer=opt, loss='mse')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=50, min_lr=0.000001, verbose=1)
checkpointer = ModelCheckpoint(monitor='val_loss', filepath="generalae.hdf5", verbose=1, save_best_only=True)

autoencoder.fit(X_train, X_train,
                epochs=500,
                batch_size=128,
                shuffle=True,
                validation_data=(X_test, X_test),
                callbacks=[reduce_lr, checkpointer])



### CHECKING ERROR LEVEL IN BOTH TYPES OF DATA ###

pred = autoencoder.predict(X_clean)
pred_bad = autoencoder.predict(X_bad_filtered)


X_test, pred = X_clean, pred
good_error, good_var = 0., 0.
for x, p in zip(X_test, pred):
	good_error += np.mean(np.abs(x - p))
	good_var += np.std(np.abs(x - p))
good_error /= len(X_test)
good_var /= len(X_test)
print good_error, good_var

X_test, pred = X_bad_filtered, pred_bad
good_error, good_var = 0., 0.
for x, p in zip(X_test, pred):
	good_error += np.mean(np.abs(x - p))
	good_var += np.std(np.abs(x - p))
good_error /= len(X_test)
good_var /= len(X_test)
print good_error, good_var


### CHECKING ACCURACIES ###


X1, X2 = X_clean, X_bad_filtered
X1r = autoencoder.predict(X1)
X2r = autoencoder.predict(X2)
errors1 = []
for x1, x1r in zip(X1, X1r):
	err = np.abs(x1 - x1r).mean()
	errors1.append(err)

good_error = np.array(errors1).mean()
good_var = np.array(errors1).std()

for i in np.linspace(0, 3, num=30):
	correct = 0.
	tresh = good_var * i

	for x1, x1r in zip(X1, X1r):
		err = np.abs(x1 - x1r).mean()
		if err <= good_error + tresh:
			correct += 1.

	correct2 = 0.
	for x2, x2r in zip(X2, X2r):
		err = np.abs(x2 - x2r).mean()
		if err > good_error + tresh:
			correct2 += 1.

	precision = correct / len(X1)
	recall = correct2 / len(X2)

	F = 2*precision*recall / (precision + recall)

	print good_error + tresh, err
	print i, precision, recall, F
	print '-' * 20
