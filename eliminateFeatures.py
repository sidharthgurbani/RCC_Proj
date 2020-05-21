from sklearn import svm
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, RFE
from sklearn.preprocessing import StandardScaler
from openpyxl import *
import numpy as np
import copy


def updateFeatureList(feature_list, ranks, X, name):
    wb = load_workbook("Dataset/temp.xlsx")
    ws = wb.create_sheet(name)
    new_list = list()
    filename = 'Dataset/temp.xlsx'
    #wcell1 = ws.cell(1, 1, "Feature")
    #wcell2 = ws.cell(1, 2, "Rank")
    col=1
    #dict = {}
    for i,r in enumerate(ranks):
        if (r == 1):
            new_list.append(feature_list[i])
            wcell1 = ws.cell(1, col, feature_list[i])
            for row, val in enumerate(X[:,i]):
                wcell2 = ws.cell(row+2, col ,val)
            col += 1

    wb.save(filename)
    return new_list

def eliminateFeaturesRecursively(dataset, X_train, X_test, y_train, y_test, feature_list, clfname):
    if dataset=='pv':
        features = [450, 400, 350, 300, 250, 200, 150, 100, 50]
    elif dataset=='noncon':
        features = [300, 250,200, 150, 100, 50]
    list_temp = feature_list
    scaler = StandardScaler()
    X_train_minmax = scaler.fit_transform(X_train)
    X_test_minmax = scaler.transform(X_test)
    ranks = []
    scores = []
    for i,feat in enumerate(features):
        if clfname=='svc':
            clf = svm.LinearSVC()
        elif clfname=='rf':
            clf = RandomForestClassifier(n_estimators=50, max_depth=20)

        rfe = RFE(estimator=clf, step=1, n_features_to_select=feat)

        X_train_transformed = rfe.fit_transform(X_train_minmax, y_train)
        X_test_transformed = rfe.transform(X_test_minmax)
        score = rfe.score(X_test_minmax, y_test)
        scores.append(score)
        print("For number of features = {}\n".format(feat))
        print("Shape of transformed train dataset is: {}".format(X_train_transformed.shape))
        print("Optimal no. of features are: {}".format(rfe.n_features_))
        print("Mean Score of transformed dataset with CV is: {:.2f}".format(score))
        ranks.append(rfe.ranking_)
        name = "Top" + str(feat)
        list_temp = updateFeatureList(list_temp, rfe.ranking_, X_train_minmax, name)
        print("Shape of ranks is: {}\n\n".format(ranks[i].shape))
        X_train_minmax = copy.deepcopy(X_train_transformed)
        X_test_minmax = copy.deepcopy(X_test_transformed)

    return X_train_minmax, X_test_minmax

def eliminateFeaturesRecursivelyWithCV(X, y, clfname, feature_list=None):
    # list_temp = feature_list
    scaler = StandardScaler()
    X_minmax = scaler.fit_transform(X)
    scores = []
    if clfname == 'svc':
        clf = svm.LinearSVC()
    elif clfname == 'rf':
        clf = RandomForestClassifier(n_estimators=50, max_depth=10)

    kfold = StratifiedKFold(n_splits=5, shuffle=True)

    for outer in range(1,3):
        print("\n--------This is outer loop {}---------\n".format(outer))
        for i,(train_o, test_o) in enumerate(kfold.split(X_minmax, y)):
            print("This is set {}".format(i+1))
            X_train_o = X_minmax[train_o]
            y_train_o = y[train_o]
            X_test_o = X_minmax[test_o]
            y_test_o = y[test_o]
            X_train_transformed = copy.deepcopy(X_train_o)
            X_test_transformed = copy.deepcopy(X_test_o)

            for inner in range(1,4):
                n_feat = min(100, X_train_transformed.shape[1])
                print("\n\t--------This is inner loop {}---------\n".format(inner))
                rfecv = RFECV(estimator=clf, step=1, min_features_to_select=n_feat, cv=kfold, scoring='roc_auc')
                X_train_transformed = rfecv.fit_transform(X_train_transformed, y_train_o)
                X_test_transformed = rfecv.transform(X_test_transformed)
                X_minmax = rfecv.transform(X_minmax)
                features = rfecv.n_features_
                ranking = rfecv.ranking_
                print("\tShape of transformed train dataset is: {}".format(X_train_transformed.shape))
                print("\tOptimal no. of features are: {}".format(features))
                # name = "Top" + str(features)
                # list_temp = updateFeatureList(list_temp, ranking, X_minmax, name)

            n_feat = max(5,X_train_transformed.shape[1] - 10)
            rfe = RFE(estimator=clf, step=1, n_features_to_select=n_feat)
            X_train_transformed = rfe.fit_transform(X_train_transformed, y_train_o)
            score = rfe.score(X_test_transformed, y_test_o)
            X_test_transformed = rfe.transform(X_test_transformed)
            X_minmax = rfe.transform(X_minmax)
            scores.append(score)
            print("\nNumber of features selected are: {}".format(n_feat))
            print("Shape of transformed train dataset is: {}".format(X_train_transformed.shape))
            print("Score for this loop is: {}".format(score))
            ranking = rfe.ranking_
            # name = "Top" + str(feat)
            # list_temp = updateFeatureList(list_temp, ranking, X_minmax, name)
            print("Shape of ranks is: {}\n\n".format(ranking.shape))

    #print("After outer loop CV, mean score is: {}".format(scores.mean()))
    X_final = np.vstack((X_train_transformed, X_test_transformed))

    return X_final

def noRFE(X, y, clfname, scale=False):
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    if clfname=='svc':
        clf = svm.LinearSVC()
    elif clfname=='rf':
        clf = RandomForestClassifier(n_estimators=10, max_depth=20)

    score = cross_val_score(clf, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True))
    print("{} Classifier gives mean accuracy after CV {:.2f}".format(clfname, score.mean() * 100))
