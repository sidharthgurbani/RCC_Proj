from sklearn import svm
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV, RFE
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
from openpyxl import *
import pandas as pd

def featureSelectAndClassifyRFECV(X_train, X_test, y_train, y_test):

    scaler = MinMaxScaler()
    #scaler = StandardScaler()
    #scaler = RobustScaler()
    X_train_minmax = scaler.fit_transform(X_train)
    X_test_minmax = scaler.transform(X_test)

    #svc =svm.LinearSVC()
    rf = RandomForestClassifier(n_estimators=50, max_depth=20)

    rfecv = RFECV(estimator=rf, step=1, min_features_to_select=5, cv = StratifiedKFold(5), scoring='accuracy')

    X_train_transformed = rfecv.fit_transform(X_train_minmax, y_train)
    #X_train_transformed = rfecv.fit_transform(X_train, y_train)
    X_test_transformed = rfecv.transform(X_test_minmax)
    #X_test_transformed = rfecv.transform(X_test)
    score = rfecv.score(X_test_minmax, y_test)
    #score = rfecv.score(X_test, y_test)

    print('Optimal no. of features are ' + str(rfecv.n_features_))
    print('Score for test set is ' + str(score))
    print(rfecv.ranking_.shape)
    print(X_train_transformed.shape)
    print(X_test_transformed.shape)

    plt.figure()
    plt.xlabel('no. of features')
    plt.ylabel('cv score')
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()


def featureSelectAndClassifyRFE(X_train, X_test, y_train, y_test):

    scaler = MinMaxScaler()
    X_train_minmax = scaler.fit_transform(X_train)
    #X_test_minmax = scaler.transform(X_test)

    ranks = []
    for i in range(1, 2):
        #svc =svm.LinearSVC()
        rf = RandomForestClassifier(n_estimators=20, max_depth=15)
        rfe = RFE(estimator=rf, step=1, n_features_to_select=300)
        X_train_transformed = rfe.fit_transform(X_train_minmax, y_train)
        print(X_train_transformed.shape)
        #X_test_transformed = rfe.transform(X_test_minmax)

        #score = rfe.score(X_test_minmax, y_test)

        print('Optimal no. of features are ' + str(rfe.n_features_))
        #print('Score for test set is ' + str(score))
        ranks.append(rfe.ranking_)

    print(ranks[0].shape)
    print(ranks[0])
    return ranks[0]


    #print(rfe.ranking_)
    #print(X_train_transformed.shape)
    #print(X_test_transformed.shape)

def writeToExcelFile(ranks):
    wb =load_workbook("Dataset/temp.xlsx")
    ws = wb.create_sheet("Sheet2")
    i=5
    for val in ranks:
        wcell = ws.cell(5,i)
        wcell.value = val
        i=i+1

    wb.save("Dataset/temp.xlsx")

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
    #scaler = MinMaxScaler()
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
        print("Score of transformed test dataset is: {:.2f}".format(score))
        ranks.append(rfe.ranking_)
        name = "Top" + str(feat)
        list_temp = updateFeatureList(list_temp, rfe.ranking_, X_train_minmax, name)
        print("Shape of ranks is: {}\n\n".format(ranks[i].shape))
        X_train_minmax = X_train_transformed
        X_test_minmax = X_test_transformed

    return X_train_minmax, X_test_minmax

def noRFE(X, y, clfname, scale=False):
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    if clfname=='svc':
        clf = svm.LinearSVC()
    elif clfname=='rf':
        clf = RandomForestClassifier(n_estimators=10, max_depth=20)

    score = cross_val_score(clf, X, y, cv=StratifiedKFold(5))
    print("{} Classifier gives mean accuracy after CV {:.2f}".format(clfname, score.mean() * 100))