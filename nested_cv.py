from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, check_cv, cross_val_score
from sklearn.base import clone, is_classifier
from sklearn.utils import indexable
from boost import CVscores
from math import sqrt
import copy
from imblearn.metrics import geometric_mean_score
from statistics import mean, stdev
from random import randint
import pandas as pd
import matplotlib.pyplot as plt


def feature_selection(dataset, model, df, y, sorted_feature_list, th=0.01):
    list1 = []
    best_score = 0
    best_list = []
    cv_scores = []

    for i, (key, _) in enumerate(sorted_feature_list):
        list1.append(key)
        X = df[list1].to_numpy()
        cv_score = []
        for _ in range(20):
            rs = randint(1, 100)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
            cv_score.append(mean(cross_val_score(clone(model), X=X, y=y, scoring='accuracy', cv=cv)))

        mean_score = mean(cv_score)
        # print(len(cv_score))
        # print(type(mean_score))
        cv_scores.append(mean_score)
        if mean_score > (best_score + th):
            best_score = mean_score
            best_list = copy.deepcopy(list1)

    print("Best score is {} and number of features are {}".format(best_score, len(best_list)))

    return cv_scores, best_list


def feature_assessment_and_selection(dataset, model, X, y, df, feature_list):
    features = run_cv(model=clone(model), X=X, y=y, feature_list=feature_list)
    sorted_feature_list = sort_features(dataset=dataset, features=features, feature_list=feature_list)

    cv_scores, best_features = feature_selection(dataset=dataset, model=clone(model), df=df, y=y,
                                                 sorted_feature_list=sorted_feature_list)

    print("Best features are {}".format(best_features))
    features_df = pd.DataFrame(best_features)
    features_df.to_excel("Dataset/" + dataset + "best_features.xlsx")

    plt.figure()
    plt.xlabel("Number of features selected")

    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(cv_scores) + 1), cv_scores)
    plt.savefig("Dataset/" + dataset + "_CVscores_vs_numberOfFeatures.jpg")
    plt.show()

    return best_features


def get_feature_weights(estimator, X, y, feature_list, features, impute, feature_select,
                              runs, diag=False):
    scores = {
        "acc_scores": [],
        "f1_scores": [],
        "p_scores": [],
        "r_scores": [],
        "auc_scores": [],
        "geometric_mean_scores": []
    }

    for _ in range(runs):
        rs = randint(1, 100)
        mean_score = CVscores(clone(estimator), X, y, features, feature_list, impute=impute,
                              feature_select=feature_select, rs=rs)
        [scores[key].append(mean_score[key]) for key in mean_score]

    means = {}
    stderrs = {}
    for key in scores:
        means[key] = mean(scores[key])
        stderrs[key] = stdev((scores[key]))/sqrt(runs)
        if diag==True:
            print("After {} iterations, {} mean score is {:.2f} and std error is {}".format(
                runs, key, means[key], stderrs[key]))

    return


def sort_features(dataset, features, feature_list, save_file=False):
    final_features = {
        'mas': {k: mean(v) for k, v in features['mas'].items() if v},
        'tas': {k: mean(v) for k, v in features['tas'].items() if v},
        'ranks': features['ranks']
    }
    final_features_attributes = {}
    for key in feature_list:
        final_features_attributes[key] = [
            final_features['tas'][key] if key in final_features['tas'] else 0,
            final_features['mas'][key] if key in final_features['mas'] else 0,
            final_features['ranks'][key] if key in final_features['ranks'] else 0
        ]

    print("Final features are {}".format(final_features_attributes))
    if save_file:
        feature_df = pd.DataFrame.from_dict(final_features_attributes, orient='index', columns=['tas', 'mas', 'ranks'])
        feature_df.to_excel("Dataset/" + dataset + "feature_weights.xlsx")

    sorted_feature_list = sorted(final_features['tas'].items(), key=lambda x: x[1], reverse=True)

    return sorted_feature_list


def run_cv(model, X, y, feature_list, impute=False, feature_select=True):
    feature_rank = {}
    measured_avg_score = {}
    total_avg_score = {}

    for key in feature_list:
        feature_rank[key] = 0
        measured_avg_score[key] = []
        total_avg_score[key] = []

    features = {
        'ranks': feature_rank,
        'mas': measured_avg_score,
        'tas': total_avg_score
    }

    print("Shape of X is :{}".format(X.shape))
    print("Length of feature is : {}".format(len(feature_list)))

    get_feature_weights(clone(model), X, y, feature_list, features, impute=impute,
                        feature_select=feature_select, runs=20, diag=False)

    return features


def nested_cross_validation(dataset, model, X, y, df, feature_list, impute, feature_select):
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    X, y, _ = indexable(X, y, None)
    cv = check_cv(cv, y, classifier=is_classifier(model))
    mean_scores = {}
    scores = {
        "acc_scores": [],
        "f1_scores": [],
        "p_scores": [],
        "r_scores": [],
        "auc_scores": [],
        "geometric_mean_scores": []
    }
    for train, test in cv.split(X, y):
        X_train = X[train]
        y_train = y[train]
        best_features = feature_assessment_and_selection(dataset=dataset, model=clone(model), X=X_train, y=y_train,
                                                         df=df, feature_list=feature_list)
        X_train_reduced = df[best_features].to_numpy()[train]
        X_test = df[best_features].to_numpy()[test]
        y_test = y[test]
        estimator = clone(model)
        estimator.fit(X_train_reduced, y_train)
        y_pred =estimator.predict(X_test)
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

    print(mean_scores)

    return