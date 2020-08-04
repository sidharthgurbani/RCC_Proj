from sklearn import metrics
from imblearn.metrics import geometric_mean_score
import xgboost as xgb
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import MinMaxScaler
from imputer import impute_data
from statistics import mean, stdev
from sklearn.model_selection import StratifiedKFold, check_cv
from sklearn.utils import indexable
from sklearn.base import is_classifier, clone


def CVscores(estimator, X, y, features, feature_list, impute=True, feature_select=True, rs=33, diag=False):
    scores = {}
    mean_scores = {}
    scores["acc_scores"] = []
    scores["f1_scores"] = []
    scores["p_scores"] = []
    scores["r_scores"] = []
    scores["auc_scores"] = []
    scores["geometric_mean_scores"] = []

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
    X, y, _ = indexable(X, y, None)
    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    for train, test in cv.split(X, y):

        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]

        if impute:
            X_train, X_test = impute_data(X_train=X_train, X_test=X_test)

        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # model = RFECV(estimator=clone(estimator), cv=5)

        model = clone(estimator)
        model.fit(X_train, y_train)
        if feature_select:
            feature_importance = model.feature_importances_
            getImportantFeatures(feature_list, feature_importance, features)

        y_pred = model.predict(X_test)

        # Xtr_temp, _ = updateList(feature_importance, X_train, feature_list)
        # model2 = clone(estimator)
        # model2.fit(Xtr_temp, y_train)
        # Xts_temp, _ = updateList(feature_importance, X_test, feature_list)
        # y_pred = model2.predict(Xts_temp)

        acc_score = metrics.accuracy_score(y_pred=y_pred, y_true=y_test)
        auc_score = metrics.roc_auc_score(y_score=y_pred, y_true=y_test)
        r_score = metrics.recall_score(y_pred=y_pred, y_true=y_test)
        p_score = metrics.precision_score(y_pred=y_pred, y_true=y_test)
        f1_score = metrics.f1_score(y_pred=y_pred, y_true=y_test)
        gmean_score = geometric_mean_score(y_test, y_pred)

        scores["acc_scores"].append(acc_score)
        scores["auc_scores"].append(auc_score)
        scores["r_scores"].append(r_score)
        scores["p_scores"].append(p_score)
        scores["f1_scores"].append(f1_score)
        scores["geometric_mean_scores"].append(gmean_score)

    mean_scores["acc_scores"] = mean(scores["acc_scores"])
    mean_scores["auc_scores"] = mean(scores["auc_scores"])
    mean_scores["r_scores"] = mean(scores["r_scores"])
    mean_scores["p_scores"] = mean(scores["p_scores"])
    mean_scores["f1_scores"] = mean(scores["f1_scores"])
    mean_scores["geometric_mean_scores"] = mean(scores["geometric_mean_scores"])


    """ Print scores if diagnostics is set to TRUE """

    if diag==True:
        print("Original ACC scores are: {}".format(scores["acc_scores"]))
        print("Mean ACC score is : {}".format(mean_scores["acc_scores"]))

        print("Original AUC scores are: {}".format(scores["auc_scores"]))
        print("Mean AUC score is : {}".format(mean_scores["auc_scores"]))

        print("Original Recall scores are: {}".format(scores["r_scores"]))
        print("Mean Recall score is : {}".format(mean_scores["r_scores"]))

        print("Original Precision scores are: {}".format(scores["p_scores"]))
        print("Mean Precision score is : {}".format(mean_scores["p_scores"]))

        print("Original F1 scores are: {}".format(scores["f1_scores"]))
        print("Mean F1 score is : {}".format(mean_scores["f1_scores"]))

        print("Original Gmean scores are: {}".format(scores["geometric_mean_scores"]))
        print("Mean Gmean score is : {}".format(mean_scores["geometric_mean_scores"]))

    return mean_scores

def updateList(feature_importance, X, feature_list):
    indices = []
    for i, val in enumerate(feature_importance):
        if val > 0.01:
            indices.append(i)

    X_new = X[:, indices]
    # features = feature_list[indices]
    return X_new, feature_list

def getImportantFeatures(feature_list, feature_importance, features):

    for i, val in enumerate(feature_importance):
        features['tas'][feature_list[i]].append(val)
        if val > 0:
            features['ranks'][feature_list[i]] += 1
            features['mas'][feature_list[i]].append(val)

    return