import numpy as np
import typing
from collections import defaultdict


def kfold_split(num_objects: int,
                num_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    a = num_objects // num_folds
    ind = [i for i in range(num_objects)]
    list_of_length_num_folds = []
    for i in range(num_folds - 1):
        tmp = np.array(ind[0: i*a] + ind[(i + 1)*a:])
        list_of_length_num_folds.append((tmp, np.array(ind[i*a: (i + 1)*a])))
    else:
        tmp = (np.array(ind[0: a*(i + 1)]), np.array(ind[a*(i + 1):]))
        list_of_length_num_folds.append(tmp)
    return list_of_length_num_folds


def knn_cv_score(X: np.ndarray, y: np.ndarray, parameters: dict[str, list],
                 score_function: callable,
                 folds: list[tuple[np.ndarray, np.ndarray]],
                 knn_class: object) -> dict[str, float]:
    dict = {}
    for normalizer in parameters['normalizers']:
        for n_neighbors in parameters['n_neighbors']:
            for metrics in parameters["metrics"]:
                for weights in parameters["weights"]:
                    scores = []
                    for fold in folds:
                        X_train, y_train = X[fold[0]], y[fold[0]]
                        X_test, y_test = X[fold[1]], y[fold[1]]
                        if normalizer[0]:
                            scaler = normalizer[0]
                            scaler.fit(X_train)
                            X_train = scaler.transform(X_train)
                            X_test = scaler.transform(X_test)
                        neigh = knn_class(n_neighbors=n_neighbors,
                                          weights=weights,
                                          metric=metrics)
                        neigh.fit(X_train, y_train)
                        y_predict = neigh.predict(X_test)
                        scores.append(score_function(y_test, y_predict))
                    value = np.mean(scores)
                    tmp = (normalizer[1], n_neighbors, metrics, weights)
                    dict[tmp] = value
    return dict