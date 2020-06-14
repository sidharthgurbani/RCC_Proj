from sklearn import svm
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, permutation_test_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import models

# This runs Random Forest on the dataset and gives a CV score average.
def noRFE(X, y, scale=False):
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=10, max_depth=20)
    score = cross_val_score(clf, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True))
    print("RF Classifier gives mean accuracy after CV {:.2f}%".format(score.mean() * 100))

def justRF_temp(X, y, feature_list):
    model = models.RF(feature_list)
    score = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True))
    print("Final sore with just RF is {:.2f}%".format(score.mean() * 100))


def nestedCV_temp(X, y, feature_list):
    model = models.nestedRFECV(feature_list)
    score = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True))
    print("Final sore with just RF is {:.2f}%".format(score.mean() * 100))

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
    model1.fit(X,y)
    score1 = model1.score(y)
    print("Final sore with just RF is {:.2f}%".format(score1.mean() * 100))
    print("\nUsing nested cross-validation on these filtered features and checking the accuracy\n")
    model2.fit(X,y)
    score2 = model2.score(y)
    print("Final sore with nested CV is {:.2f}%".format(score2.mean() * 100))

def permutationTest(X, y, model):
    #model = RandomForestClassifier(n_estimators=10, max_depth=20)
    score, permutation_scores, pvalue = permutation_test_score(model, X, y, scoring='accuracy',
                                                               cv=StratifiedKFold(n_splits=5, shuffle=True))
    print("Score is {} and pvalue is {}".format(score, pvalue))
    print(permutation_scores)
