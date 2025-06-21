from catboost import CatBoostRegressor
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from numpy import ndarray


key_dict = {}
gen = []
direc = []
loc = []


def transform(X, flag_of_train):
    global key_dict, gen, direc, loc
    if flag_of_train:
        key_arr = [j for i in list(X['keywords']) for j in i]
        key_dict = {}
        for i in key_arr:
            if i in key_dict.keys():
                key_dict[i] += 1
            else:
                key_dict[i] = 1
        key_dict = dict(sorted(key_dict.items(), key=lambda x: x[1], reverse=True))
        gen = list(X['genres'])
        direc = list(X['directors'])
        loc = list(X['filming_locations'])

    for i in key_dict.items():
        if i[1] < 50:
            break
        else:
            X[i[0]] = 0
    for val in set(i for j in list(X['keywords']) for i in j):
        if val in key_dict.keys() and key_dict[val] >= 50:
            X.loc[[val in y['keywords'] for x, y in X.iterrows()], val] = 1
    for w in [(gen, 'genres'), (direc, 'directors'), (loc, 'filming_locations')]:
        X[[i for arr in w[0] for i in arr]] = 0
        for val in set(i for j in w[0] for i in j):
            X.loc[[val in y[w[1]] for x, y in X.iterrows()], val] = 1
        X.drop(columns=w[1], inplace=True)
    for i in [0, 1, 2]:
        X[f'actor_{i}_gender'] = X[f'actor_{i}_gender'].astype('category')

    return X.drop(columns='keywords')


def train_model_and_predict(train_file: str, test_file: str) -> ndarray:
    """
    This function reads dataset stored in the folder, trains predictor and returns predictions.
    :param train_file: the path to the training dataset
    :param test_file: the path to the testing dataset
    :return: predictions for the test file in the order of the file lines (ndarray of shape (n_samples,))
    """
    df_train = pd.read_json(train_file, lines=True)
    df_test = pd.read_json(test_file, lines=True)
    y_train = df_train['awards']
    df_train.drop(columns='awards', inplace=True)
    X_train, X_test = transform(df_train, True), transform(df_test, False)
    cats = ['actor_0_gender', 'actor_1_gender', 'actor_2_gender']
    params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 6
    }
    reg = CatBoostRegressor(**params, cat_features=cats, train_dir='/tmp/catboost_info', verbose=False)
    reg.fit(X_train, y_train)
    return reg.predict(X_test)