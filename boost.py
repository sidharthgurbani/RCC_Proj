import models
from sklearn import metrics
import xgboost as xgb
from statistics import mean
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import indexable
from sklearn.model_selection._split import check_cv
from sklearn.base import is_classifier, clone

def XGboost(X, y, feature_list):
    scores = []
    scores_tr = []
    # corr = models.CorrMatrix()
    # corr.fit(X, y)
    # X, y = corr.transform()
    print(X.shape)
    print(y.shape)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=33)
    X, y, _ = indexable(X, y, None)
    estimator = xgb.XGBClassifier()
    estimator_tr = xgb.XGBClassifier()
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    final_features = set([])
    for train, test in cv.split(X, y):

        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]

        model = clone(estimator)
        model.fit(X_train, y_train)

        feature_importance = model.feature_importances_
        X_tr_train, features = updateList(feature_importance, X_train, feature_list)
        X_tr_test, _ = updateList(feature_importance, X_test, feature_list)

        # print(len(features))
        for feat in features:
            final_features.add(feat)

        model_tr = clone(estimator_tr)
        model_tr.fit(X_tr_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_tr = model_tr.predict(X_tr_test)

        # score = metrics.accuracy_score(y_pred=y_pred, y_true=y_test)
        # score_tr = metrics.accuracy_score(y_pred=y_pred_tr, y_true=y_test)

        score = metrics.roc_auc_score(y_true=y_test, y_score=y_pred)
        score_tr = metrics.roc_auc_score(y_true=y_test, y_score=y_pred_tr)

        # score = metrics.recall_score(y_pred=y_pred, y_true=y_test)
        # score_tr = metrics.recall_score(y_pred=y_pred_tr, y_true=y_test)

        scores.append(score)
        scores_tr.append(score_tr)

    print("Original scores are: {}".format(scores))
    print("Transformed scores are: {}".format(scores_tr))
    print("Mean scores are {:.2f} and {:.2f}".format(mean(scores), mean(scores_tr)))
    # print(len(final_features))
    # print(final_features)

    return mean(scores), mean(scores_tr)

def updateList(feature_importance, X, feature_list):
    indices = []
    for i, val in enumerate(feature_importance):
        if val > 0.01:
            indices.append(i)

    X_new = X[:, indices]
    features = feature_list[indices]
    return X_new, features