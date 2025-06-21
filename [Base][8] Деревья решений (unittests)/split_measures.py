import numpy as np


def evaluate_measures(sample):
    """Calculate measure of split quality (each node separately).

    Please use natural logarithm (e.g. np.log) to evaluate value of entropy measure.

    Parameters
    ----------
    sample : a list of integers. The size of the sample equals to the number of objects in the current node. The integer
    values are equal to the class labels of the objects in the node.

    Returns
    -------
    measures - a dictionary which contains three values of the split quality.
    Example of output:

    {
        'gini': 0.1,
        'entropy': 1.0,
        'error': 0.6
    }

    """
    dict_p = {}
    for i in sample:
        if i in dict_p.keys():
            dict_p[i] += 1
        else:
            dict_p[i] = 1
    p_arr = []
    for i in dict_p:
        p_arr.append(dict_p[i] / len(sample))
    p_arr = np.array(p_arr)
    gini = 1 - np.power(p_arr, 2).sum()
    error = 1 - np.max(p_arr)
    entropy = - np.dot(np.log(p_arr), p_arr)
    measures = {'gini': float(gini), 'entropy': float(entropy), 'error': float(error)}
    return measures