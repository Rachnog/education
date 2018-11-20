from keras.callbacks import Callback
from math import log
from sklearn.metrics import log_loss


class Consistency(Callback):
    """
    Callback class that calculates Numerai consistency metric at each epoch
    of training. It also adds the consistency to the training history.
    """

    def __init__(self, tournament_df):
        """
        :param tournament_df: Pandas DataFrame containing the Numerai tournament data
        """
        super(Consistency, self).__init__()
        self.era_indices = self._get_era_indices(tournament_df)

    def _get_era_indices(self, tournament_df):
        indices = []
        validation = tournament_df[tournament_df.data_type == 'validation']
        for era in validation.era.unique():
            indices.append(validation.era == era)
        return indices

    def _consistency(self, era_indices, validation_y, validation_yhat):
        num_better_random = 0.0
        for indices in era_indices:
            labels = validation_y[indices]
            era_preds = validation_yhat[indices]
            ll = log_loss(labels, era_preds)
            if ll < .693:
                num_better_random += 1.0
        return num_better_random / len(era_indices)

    def on_epoch_end(self, epoch, logs=None):
        y_hat = self.model.predict(self.validation_data[0])
        y = self.validation_data[1]
        c = self._consistency(self.era_indices, y, y_hat)
        if logs is not None:
            logs['consistency'] = c
        print('  consistency: {:.2%}'.format(c))




class ConsistencySecondLoss(Callback):
    """
    Callback class that calculates Numerai consistency metric at each epoch
    of training. It also adds the consistency to the training history.
    """

    def __init__(self, tournament_df, dataname, dataid):
        """
        :param tournament_df: Pandas DataFrame containing the Numerai tournament data
        """
        super(ConsistencySecondLoss, self).__init__()
        self.era_indices = self._get_era_indices(tournament_df)
        self.dataname =  dataname
        self.dataid = dataid

    def _get_era_indices(self, tournament_df):
        indices = []
        validation = tournament_df[tournament_df.data_type == 'validation']
        for era in validation.era.unique():
            indices.append(validation.era == era)
        return indices

    def _consistency(self, era_indices, validation_y, validation_yhat):
        num_better_random = 0.0
        for indices in era_indices:
            labels = validation_y[indices]
            era_preds = validation_yhat[indices]
            ll = log_loss(labels, era_preds)
            if ll < .693:
                num_better_random += 1.0
        return num_better_random / len(era_indices)

    def on_epoch_end(self, epoch, logs=None):
        y_hat = self.model.predict(self.validation_data[0])[self.dataid]
        y = self.validation_data[1]
        c = self._consistency(self.era_indices, y, y_hat)
        if logs is not None:
            logs['consistency'] = c
        print('  ' + str(self.dataname) + ' consistency: {:.2%}'.format(c))


import numpy as np
import matplotlib.pyplot as plt

from scipy.special import erfinv

class GaussRankScaler():

    def __init__( self ):
        self.epsilon = 0.001
        self.lower = -1 + self.epsilon
        self.upper =  1 - self.epsilon
        self.range = self.upper - self.lower

    def fit_transform( self, X ):
    
        i = np.argsort( X, axis = 0 )
        j = np.argsort( i, axis = 0 )

        assert ( j.min() == 0 ).all()
        assert ( j.max() == len( j ) - 1 ).all()
        
        j_range = len( j ) - 1
        self.divider = j_range / self.range
        
        transformed = j / self.divider
        transformed = transformed - self.upper
        transformed = erfinv( transformed )
        
        return transformed