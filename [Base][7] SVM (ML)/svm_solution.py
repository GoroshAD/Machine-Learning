import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold


def train_svm_and_predict(train_features, train_target, test_features):
    """
    train_features: np.array, (num_elements_train x num_features) - train data description, the same features and the same order as in train data
    train_target: np.array, (num_elements_train) - train data target
    test_features: np.array, (num_elements_test x num_features) -- some test data, features are in the same order as train features

    return: np.array, (num_elements_test) - test data predicted target, 1d array
    """
    std = StandardScaler()
    train_features = std.fit_transform(train_features, train_target)
    test_features = std.transform(test_features)
    kf = KFold(n_splits=5)
    arr_svc = [SVC(C=10, random_state=42),
               LogisticRegression(C=0.1, penalty='l1', solver='liblinear', random_state=42),
               DecisionTreeClassifier(random_state=42),
               DecisionTreeClassifier(criterion='entropy', random_state=42),
               SVC(C=10, kernel='poly', degree=1, random_state=42)
               ]
    pred_arr, pred = [], []
    for i, (index1, index2) in enumerate(kf.split(train_features)):
        arr_svc[i].fit(train_features[index1, :], train_target[index1])
        pred_arr.append(arr_svc[i].predict(test_features))

    pred_arr = np.array(pred_arr)
    for i in range(len(test_features)):
        uniqs, counts = np.unique(pred_arr[:, i], return_counts=True)
        if len(uniqs) > 1:
            pred.append(uniqs[0] if (counts[0] > counts[1]) else uniqs[1])
        else:
            pred.append(uniqs[0])
    """
    svmodel = SVC(C=10, kernel='rbf')
    svmodel.fit(train_features, train_target)
    pred = svmodel.predict(test_features)
    """
    return np.array(pred)