import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, permutation_test_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from statistics import mean, stdev
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
    score = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True))
    print("Final sore with just RF is {:.2f}%".format(score.mean() * 100))


def nestedCV_temp(X, y, feature_list):
    model = models.nestedRFECV(feature_list)
    model.fit(X,y)
    y_hat = model.predict(X)
    print(y_hat.shape)
    acc_score = model.score()
    X_tr = model.transformed()
    clf = RandomForestClassifier(n_estimators=10, max_depth=20)
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

def permutationTest(X, y, feature_list, dataset):
    print("\n\nCurrent dataset is " + dataset + "\n")
    scores = []
    total_permutations = []
    pvalues = []
    for i in range(5):
        model = models.RF(feature_list)
        Xt = copy.deepcopy(X)
        yt = copy.deepcopy(y)
        score, permutation_score, pvalue = permutations(model, Xt, yt, scoring='accuracy',
                                                        cv=StratifiedKFold(n_splits=5, shuffle=True),
                                                        n_permuations=100)
        scores.append(score)
        print(permutation_score.shape)
        total_permutations.append(permutation_score)
        pvalues.append(pvalue)
        print("\nPermutation {}:\n".format(i+1))
        print("p-value is {:.2f}".format(pvalue))
        print("Score is {:.2f}".format(score))

    p_mean = mean(pvalues)
    p_stdev = stdev(pvalues)
    s_mean = mean(scores)
    print("\nMean p-value is {:.2f}".format(p_mean))
    print("Stdev p-value is {:.2f}".format(p_stdev))
    print("Mean score is {:.2f}\n".format(s_mean))
    n = len(total_permutations)
    r = np.max(total_permutations) - np.min(total_permutations)
    size = sqrt(n)
    width = ceil(r/size)
    plt.hist(x=total_permutations, bins=ceil(n/width))
    plt.axvline(x=s_mean, color='r', label='Average score value')
    plt.title("Histogram plot of p-values for " + dataset)
    plt.xlabel("p-value")
    plt.ylabel("Frequency of p-values")
    plt.legend()
    plt.savefig('Histogram_' + dataset + '_.png')
    plt.show()