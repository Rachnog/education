from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cross_validation import train_test_split

import numpy as np

def make_x_y(points, images, labels, labels2):
    '''
        Prepare training pairs of images and labels
        return: X: images 50x50px
                Y1: emotions labels
                Y2: FACS labels
                Y3: facial keypoints
    '''
    X, Y = [], []
    for key in labels.keys():
        for point, image in zip(points[key], images[key])[5:-5]:
            X.append(image)
            Y.append([labels[key], labels2[key], np.array(point).flatten()])

    # X, Y = shuffle(X, Y)

    Y1 = [y[0][0] for y in Y]
    Y2 = [[yi[0] for yi in y[1]] for y in Y]
    Y3 = [y[2] for y in Y]

    mlb = MultiLabelBinarizer()
    Y1 = mlb.fit_transform(Y1)
    mlb = MultiLabelBinarizer()
    Y2 = mlb.fit_transform(Y2)

    X, Y1, Y2, Y3 = np.array(X), np.array(Y1), np.array(Y2), np.array(Y3)
    return X, Y1, Y2, Y3
    
    
def prepare_train_test(X, Y1, Y2, Y3, W, random_state = 42):
    '''
        Split into train and test sets
    '''
    X_train, X_test, Y_train, Y_test, Y_train2, Y_test2, Y_train3, Y_test3 = train_test_split(X, Y1, Y2, Y3, random_state = random_state)
    
    X_train, X_test, Y_train, Y_test, Y_train2, Y_test2, Y_train3, Y_test3 = np.array(X_train), \
                                                                          np.array(X_test), \
                                                                          np.array(Y_train), \
                                                                          np.array(Y_test), \
                                                                          np.array(Y_train2), \
                                                                          np.array(Y_test2), \
                                                                          np.array(Y_train3), \
                                                                          np.array(Y_test3)
    sc = MinMaxScaler()
    Y_train3 = sc.fit_transform(Y_train3)
    Y_test3 = sc.fit_transform(Y_test3)
    X_train = X_train.reshape(X_train.shape[0], W, W, 1)
    X_test = X_test.reshape(X_test.shape[0], W, W, 1)
    return X_train, X_test, Y_train, Y_test, Y_train2, Y_test2, Y_train3, Y_test3