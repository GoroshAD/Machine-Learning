import os

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

import numpy as np
from scipy.ndimage import shift, center_of_mass
from sklearn.model_selection import GridSearchCV


class PotentialTransformer:
    """
    A potential transformer.

    This class is used to convert the potential's 2d matrix to 1d vector of features.
    """
    def fit(self, X, Y):
        """
        Build the transformer on the training set.
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: trained transformer
        """
        return self

    def fit_transform(self, X, Y):
        """
        Build the transformer on the training set and return the transformed dataset (1d vectors).
        :param x: list of potential's 2d matrices
        :param y: target values (can be ignored)
        :return: transformed potentials (list of 1d vectors)
        """
        return self.transform(X)

    def transform(self, X):
        """
        Transform the list of potential's 2d matrices with the trained transformer.
        :param x: list of potential's 2d matrices
        :return: transformed potentials (list of 1d vectors)
        """
        tmp, tmp_p = [], []
        for image in X:
            if np.count_nonzero(image == image.min()) > 3:
                y_min, x_min = center_of_mass(20 - image)
                tmp_p = 20 - shift(20 - image, (128 - y_min, 128 - x_min), mode='grid-wrap')
            else:
                ver, hor = 128 - np.array(np.unravel_index(image.argmin(), image.shape))
                tm = np.roll(image, hor)
                tmp_p = np.roll(tm, ver, axis=0)
            tmp_p = np.mean(tmp_p, axis=1)
            tmp_p[: len(tmp_p) // 2] = sorted(tmp_p[: len(tmp_p) // 2], reverse=True)
            tmp_p[len(tmp_p) // 2:] = sorted(tmp_p[len(tmp_p) // 2:])
            tmp.append(tmp_p)
        tmp = np.stack(tmp)
        return tmp.reshape((tmp.shape[0], -1))


def load_dataset(data_dir):
    """
    Read potential dataset.

    This function reads dataset stored in the folder and returns three lists
    :param data_dir: the path to the potential dataset
    :return:
    files -- the list of file names
    np.array(X) -- the list of potential matrices (in the same order as in files)
    np.array(Y) -- the list of target value (in the same order as in files)
    """
    files, X, Y = [], [], []
    for file in sorted(os.listdir(data_dir)):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)


def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)
    # it's suggested to modify only the following line of this function
    from time import time
    start = time()
    regressor = Pipeline([('vectorizer', PotentialTransformer()), ('etr', ExtraTreesRegressor(n_estimators=200, criterion='poisson'))])
    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)
    print(time() - start)
    return {file: value for file, value in zip(test_files, predictions)}