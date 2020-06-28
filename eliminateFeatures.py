import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, permutation_test_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
import pandas as pd
import xgboost as xgb
from math import sqrt, ceil
import numpy as np
import copy
import models
from permutation import permutations

# This runs Random Forest on the dataset and gives a CV score average.
def noRFE(X, y, scale=False):
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=100, max_depth=50)
    score = cross_val_score(clf, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True))
    print("RF Classifier gives mean accuracy after CV {:.2f}%".format(score.mean() * 100))

def justRF_temp(X, y, feature_list):
    model = models.RF(feature_list)
    model1 = models.RF(feature_list)
    model1.fit(X,y)
    X_tr, _ = model1.updateList(X)
    print(X_tr.shape)
    score = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True))
    print("Final sore with just RF is {:.2f}%".format(score.mean() * 100))

def justXgBoost(X, y):
    model = xgb.XGBClassifier()
    score = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True))
    print("XGB Classifier gives mean accuracy after CV {:.2f}%".format(score.mean() * 100))

def XGBWithFeatureElimination(X, y, feature_list):
    model = models.XGBC(feature_list)
    score = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True))
    print("XGB Classifier With FE gives mean accuracy after CV {:.2f}%".format(score.mean() * 100))

def nestedCV_temp(X, y, feature_list):
    model = models.nestedRFECV(feature_list)
    model.fit(X,y)
    y_hat = model.predict(X)
    print(y_hat.shape)
    acc_score = model.score()
    X_tr = model.transformed()
    # clf = RandomForestClassifier(n_estimators=10, max_depth=20)
    clf = xgb.XGBClassifier()
    score = cross_val_score(clf, X_tr, y, cv=StratifiedKFold(n_splits=5, shuffle=True))
    print("Final sore with nested CV is {:.2f}%".format(score.mean() * 100))
    print("Final sore with nested CV is {:.2f}%".format(acc_score * 100))

def pearson_temp(df, dataset, target):
    model = models.PearsonCorr(df, dataset, target)
    print("\n Obtaining Pearson Correalation Matrix\n")
    model.getPearsonCorrelation()
    print("\n Filtering out highly correlated features\n")
    model.filterHighlyCorrelatedFeatures()
    return model.transformed()

def getFeatImportanceFromPermutation(fit, X, y):
    result = permutation_importance(fit, X, y)
    print("\n Importance mean of features are\n {}".format(result.importances_mean))
    print("\n Importance std dev of features are\n {}".format(result.importances_std))

def runPostFiltering(X, y, feature_list):
    print("\nGetting accuracy of dataset after filtering these features\n")
    noRFE(X, y)
    model1 = models.RF(feature_list)
    model2 = models.nestedRFECV(feature_list)
    print("\nGetting the weights of these filtered features and checking the accuracy\n")
    score1 = cross_val_score(model1, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True))
    print("Final sore with just RF is {:.2f}%".format(score1.mean() * 100))
    print("\nUsing nested cross-validation on these filtered features and checking the accuracy\n")
    model2.fit(X,y)
    X_tr = model2.transformed()
    clf = RandomForestClassifier(n_estimators=10, max_depth=20)
    score2 = cross_val_score(clf, X_tr, y, cv=StratifiedKFold(n_splits=5, shuffle=True))
    print("Final sore with nested CV is {:.2f}%".format(score2.mean() * 100))

def removeFeaturesWithNan():
    dataset = "Dataset/Orignial files/temp_pv.xlsx"
    df = pd.read_excel(dataset)
    X = df.drop(["Case", "sarc"], axis=1).to_numpy()
    y = df["sarc"].to_numpy()
    target = "sarc"
    feature_list = df.columns[2:]
    print(X.shape)
    indices = []
    for i in range(X.shape[1]):
        col = X[:,i]
        if np.isnan(np.sum(col))==True:
            indices.append(i)

    X_final = np.delete(arr=X, obj=indices, axis=1)
    # justRF_temp(X_final, y, feature_list)
    return X_final, y, feature_list


def permutationTest(X, y, feature_list, dataset):
    n_tests = 2
    n_permutations = 100
    scores = np.zeros(n_tests)
    pvalues = np.zeros(n_tests)
    permutation_scores = np.zeros(n_tests*n_permutations)
    for i in range(n_tests):
        # model = models.RF(feature_list)
        model = xgb.XGBClassifier()
        Xt = copy.deepcopy(X)
        yt = copy.deepcopy(y)
        score, permutation_score, pvalue = permutations(model, Xt, yt, scoring='accuracy',
                                                        cv=StratifiedKFold(n_splits=5, shuffle=True),
                                                        n_permuations=n_permutations)
        print("\nPermutation {}:\n".format(i+1))
        print("p-value is {:.2f}".format(pvalue))
        print("Score is {:.2f}".format(score))
        scores[i] = score
        pvalues[i] = pvalue
        permutation_scores[n_permutations*i:n_permutations*(i+1)] = permutation_score

    p_mean = np.mean(pvalues)
    p_stdev = np.std(pvalues)
    s_mean = np.mean(scores)
    print("\nMean p-value is {:.2f}".format(p_mean))
    print("Stdev p-value is {:.2f}".format(p_stdev))
    print("Mean score is {:.2f}\n".format(s_mean))
    r = np.amax(permutation_scores) - np.amin(permutation_scores)
    width = 0.03
    print("Range is: {}".format(r))
    plt.hist(permutation_scores, bins=ceil(r/width))
    plt.axvline(x=s_mean, color='r', label='Average score value')
    plt.title("Histogram plot of permutation score values for " + dataset)
    plt.xlabel("permutation_scores")
    plt.ylabel("Frequency of permutation_scores")
    plt.legend()
    plt.savefig('Histogram_' + dataset + '_.png')
    plt.show()

def PCAFilter(dataset,feature_list, X, y):
    noRFE(X, y, scale=True)
    # transformer = Normalizer().fit(X.T)
    # X_norm = transformer.transform(X.T).T
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    #print("Shape of X_norm is : {}".format(X_norm.shape))
    pca = PCA(n_components=20, svd_solver='full')
    X_tr = pca.fit_transform(X_norm, y)
    #print("Shape of transformed is: {}".format(X_tr.shape))
    noRFE(X_tr, y)
    #print(pca.components_.shape)
    # print(pca.explained_variance_ratio_)
    #print(sum(pca.explained_variance_ratio_))
    features = set([])
    n_features = 5
    for i, arr in enumerate(pca.components_):
        temp = np.argpartition(-arr, n_features)
        res_args = temp[:n_features]
        for index in res_args:
            #print(i)
            features.add(feature_list[index])

    print(len(features))