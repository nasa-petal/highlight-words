import numpy as np
from skmultilearn.model_selection import iterative_train_test_split


def iterative_train_test_split_dataframe(X, y, test_size):
    df_index = np.expand_dims(X.index.to_numpy(), axis=1)
    df_index_y = np.expand_dims(y.index.to_numpy(), axis=1)
    X_train, y_train, X_test, y_test = iterative_train_test_split(
        df_index, df_index_y, test_size=test_size)
    X_train = X.loc[X_train[:, 0]]
    X_test = X.loc[X_test[:, 0]]
    y_train = y.loc[y_train[:, 0]]
    y_test = y.loc[y_test[:, 0]]
    return X_train, y_train, X_test, y_test