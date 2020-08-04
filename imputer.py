import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

def impute_data(X_train, X_test):
    imp = IterativeImputer()
    # imp = KNNImputer(n_neighbors=3)
    X_train = imp.fit_transform(X_train)
    X_test = imp.transform(X_test)

    return X_train, X_test