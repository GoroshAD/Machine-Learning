import numpy as np
from typing import Tuple


def sum_non_neg_diag(X: np.ndarray) -> int:
    return x.clip(0).sum() if (x := X.diagonal()).max() >= 0 else -1
    


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    x.sort()
    y.sort()
    return np.array_equal(x, y)


def max_prod_mod_3(x: np.ndarray) -> int:
    res = np.delete(x, 0) * np.delete(x, np.size(x) - 1)
    return res[res % 3 == 0].max() if any(res[res % 3 == 0]) else -1


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weighted = image * weights
    combined = np.sum(weighted, axis=2)
    return combined


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    x = np.repeat(x[:, 0], x[:, 1])
    y = np.repeat(y[:, 0], y[:, 1])
    return -1 if x.shape != y.shape else np.dot(x, y)


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    cosine_distances = np.dot(X, Y.T) / np.outer(np.linalg.norm(X, axis=1), np.linalg.norm(Y, axis=1))
    cosine_distances[np.isnan(cosine_distances)] = 1 
    return(cosine_distances)