from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, log_loss, fbeta_score

from sklearn.model_selection import train_test_split
from imblearn.metrics import classification_report_imbalanced

import pandas as pd
import numpy as np

import matplotlib.pylab as plt
from sklearn import metrics, preprocessing, linear_model


np.random.seed(777)

train = pd.read_csv('numerai_training_data.csv', header=0)
tournament = pd.read_csv('numerai_tournament_data.csv', header=0)
validation = tournament[tournament['data_type']=='validation']

train = train.drop(['id', 'data_type'], axis=1)
features = [f for f in list(train) if "feature" in f]
x_prediction = validation[features]
ids = tournament['id']


X = train[features]
Y_target_bernie = train['target_bernie']
Y_target_charles = train['target_charles']
Y_target_elizabeth= train['target_elizabeth']
Y_target_jordan = train['target_jordan']
Y_target_ken = train['target_ken']
X_train = X

Y_target_bernie_validation = validation['target_bernie']
Y_target_charles_validation = validation['target_charles']
Y_target_elizabeth_validation = validation['target_elizabeth']
Y_target_jordan_validation = validation['target_jordan']
Y_target_ken_validation = validation['target_ken']
X_val = x_prediction

from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Average, Lambda, Concatenate, GaussianNoise, Dot, Add, Multiply, AlphaDropout
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras import regularizers
from keras import losses
from keras.layers.noise import *
from keras import backend as K

from numeraicb import ConsistencySecondLoss
from clr_callback import CyclicLR

batch_size = 64

cb1 = ConsistencySecondLoss(tournament, 'bernie', 0)
cb2 = ConsistencySecondLoss(tournament, 'charles', 1)
cb3 = ConsistencySecondLoss(tournament, 'elizabeth', 2)
cb4 = ConsistencySecondLoss(tournament, 'jordan', 3)
cb5 = ConsistencySecondLoss(tournament, 'ken', 4)
ch = ModelCheckpoint('nmr_mt.hdf5', verbose=1, save_best_only=True, monitor='val_loss')
es = EarlyStopping(monitor='val_loss', patience = 15)
clr = CyclicLR(base_lr=0.00005, max_lr=0.0001,
                                step_size=2 * (len(X_train) / batch_size), mode='triangular')


inputs = Input(shape=(50,))
c = Dense(100, activation='relu')(inputs)
c = Dropout(0.1)(c)

predictions_target_bernie = Dense(1, activation='sigmoid', name = 'target_bernie')(c)
predictions_target_charles = Dense(1, activation='sigmoid', name = 'target_charles')(c)
predictions_target_elizabeth = Dense(1, activation='sigmoid', name = 'target_elizabeth')(c)
predictions_target_jordan = Dense(1, activation='sigmoid', name = 'target_jordan')(c)
predictions_target_ken = Dense(1, activation='sigmoid', name = 'target_ken')(c)

model = Model(inputs=[inputs], outputs=[
    predictions_target_bernie,
    predictions_target_charles,
    predictions_target_elizabeth,
    predictions_target_jordan,
    predictions_target_ken
  ])
model.compile(loss='binary_crossentropy', 
              optimizer=Adam(lr = 0.0001),
              # loss_weights = [1, 1, 1, 1, 1]
              loss_weights = [0.2, 0.2, 0.2, 0.2, 0.2]
              )


print model.summary()

model.fit(X_train, [
    Y_target_bernie, Y_target_charles, Y_target_elizabeth, Y_target_jordan, Y_target_ken
  ],
          validation_data = (X_val, [
              Y_target_bernie_validation, Y_target_charles_validation, Y_target_elizabeth_validation, Y_target_jordan_validation, Y_target_ken_validation
            ]),
          verbose = 1,
          epochs = 100,
          batch_size = batch_size,
          callbacks = [cb1, cb2, cb3, cb4, cb5, ch, es, clr])

model.load_weights('nmr_mt.hdf5')

pred = model.predict(X_val)[0]
print'BERNIE'
print "- LL:", metrics.log_loss(validation['target_bernie'], pred)
print "- ROC AUC:", metrics.roc_auc_score(validation['target_bernie'], pred)

pred1 = model.predict(X_val)[1]
print 'CHARLES'
print "- LL:", metrics.log_loss(validation['target_charles'], pred1)
print "- ROC AUC:", metrics.roc_auc_score(validation['target_charles'], pred1)

pred2 = model.predict(X_val)[2]
print 'ELIZABETH'
print "- LL:", metrics.log_loss(validation['target_elizabeth'], pred2)
print "- ROC AUC:", metrics.roc_auc_score(validation['target_elizabeth'], pred2)

pred3 = model.predict(X_val)[3]
print 'JORDAN'
print "- LL:", metrics.log_loss(validation['target_jordan'], pred3)
print "- ROC AUC:", metrics.roc_auc_score(validation['target_jordan'], pred3)

pred4 = model.predict(X_val)[4]
print 'KEN'
print "- LL:", metrics.log_loss(validation['target_ken'], pred4)
print "- ROC AUC:", metrics.roc_auc_score(validation['target_ken'], pred4)

x_prediction = tournament[features]
live_prediction = model.predict(x_prediction)
for i, name in enumerate(['bernie', 'charles', 'elizabeth', 'jordan', 'ken']):
  x_prediction = tournament[features]
  y_prediction = live_prediction[i]
  results_df = pd.DataFrame(data={'probability_' + name: y_prediction[:, 0]})
  joined = pd.DataFrame(ids).join(results_df)
  joined.to_csv(name + "submission.csv", index=False)

