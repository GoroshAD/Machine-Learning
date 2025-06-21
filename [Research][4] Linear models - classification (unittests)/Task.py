import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super(Preprocessor).__init__()
        self.dtype = dtype

    def fit(self, X, Y=None):
        n_obj, n_feat = X.shape
        self.arr = []
        for i in range(n_feat):
            arr_tmp = np.unique(X.iloc[:, i])
            ones_tmp = np.eye(arr_tmp.shape[0], dtype=int)
            dict_tmp = {}
            for j in range(len(arr_tmp)):
                dict_tmp[arr_tmp[j]] = ones_tmp[j]
            self.arr.append(dict_tmp)

    def transform(self, X):
        n_obj, n_feat = X.shape
        arr = []

        def fun(n):
            arr_tmp = np.array([])
            for j in range(n_feat):
                arr_tmp = np.concatenate((arr_tmp, self.arr[j][X.iloc[n, j]]))
            arr.append(arr_tmp)
        np.vectorize(fun)(np.arange(n_obj))
        return np.array(arr[1:])

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def fit(self, X, Y):
        n_obj, n_feat = X.shape
        self.arr = []
        for i in range(n_feat):
            arr_tmp, counter = np.unique(X.iloc[:, i], return_counts=True)
            dict_tmp = {}
            for j in range(len(arr_tmp)):
                success = np.sum(Y.iloc[np.where(X.iloc[:, i] == arr_tmp[j])[0]]) / counter[j]
                counters = counter[j] / n_obj
                dict_tmp[arr_tmp[j]] = np.array([success, counters])
            self.arr.append(dict_tmp)

    def transform(self, X, a=1e-5, b=1e-5):
        n_obj, n_feat = X.shape
        arr_fin = []
        for i in range(n_obj):
            arr_tmp = np.array([])
            for j in range(n_feat):
                success = self.arr[j][X.iloc[i, j]][0]
                counter = self.arr[j][X.iloc[i, j]][1]
                arr_tmp = np.concatenate((arr_tmp, np.array([success, counter, (success + a) / (counter + b)])))
            arr_fin.append(arr_tmp)
        return np.array(arr_fin)

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:
    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds

    def fit(self, X, Y, seed=1):
        n_obj = X.shape[0]
        self.scencoder = [SimpleCounterEncoder() for i in range(self.n_folds)]
        self.folds = [*group_k_fold(n_obj, self.n_folds, seed)]
        self.arr = []
        for i in range(self.n_folds):
            self.arr.append(self.scencoder[i].fit(X.iloc[self.folds[i][1]],
                            Y.iloc[self.folds[i][1]]))
        self.arr = np.array(self.arr)

    def transform(self, X, a=1e-5, b=1e-5):
        n_obj, n_feat = X.shape
        arr_fin = np.array([[0]*3*n_feat for i in range(n_obj)], dtype=self.dtype)
        for i in range(self.n_folds):
            tmp = self.scencoder[i].transform(X.iloc[self.folds[i][0]], a, b)
            arr_fin[self.folds[i][0]] = tmp
        return arr_fin


def weights(x, y):
    return np.array([np.sum(y[np.where(x == np.unique(x)[i])]) / len(y[np.where(x == np.unique(x)[i])]) for i in range(len(np.unique(x)))])