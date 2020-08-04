from featureSelectAndClassify import featureSelectAndClassifyRFE, featureSelectAndClassifyRFECV, writeToExcelFile
from eliminateFeatures import *
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from nested_cv import nested_cross_validation, feature_assessment_and_selection
import warnings
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from statistics import mean, stdev
import numpy as np
import xgboost as xgb
from random import randint
from sklearn.base import clone

warnings.filterwarnings("ignore")


def get_dataset(dataset, path="Dataset/Original files/"):
    # Choose the dataset accordingly and set the X,y and feature values
    print("\n Dataset is: {}\n".format(dataset))

    X = None
    y = None
    feature_list = None
    df = None
    target = None

    drop_list_sarc = ["Case", "sarc"]
    drop_list_fgrade = ["Case", "rcctype", "fgrade", "perineph", "recurr", "fucond", "deceased", "survival"]
    drop_list_time = ["STUDY_DATE_DAY", "STUDY_DATE_MONTH", "STUDY_DATE_TIME", "STUDY_DATE_YEAR",
                      "TIME_POINT_RELATIVE","TIME_STAMP_DAY", "TIME_STAMP_MONTH", "TIME_STAMP_TIME",
                      "TIME_STAMP_YEAR"]

    if dataset == 'noncon_sarc':
        # df = pd.read_excel(path+"temp_noncon_imp.xlsx")
        df = pd.read_excel(path+"temp_noncon.xlsx")
        drop_list = drop_list_sarc + drop_list_time
        X = df.drop(drop_list, axis=1).to_numpy()
        y = df["sarc"].to_numpy()
        target = "sarc"
        feature_list = [x for x in df.columns if x not in drop_list]
        feature_list = np.array(feature_list)

    elif dataset == 'pv_sarc':
        # df = pd.read_excel(path+"temp_pv_imp.xlsx")
        df = pd.read_excel(path + "temp_pv.xlsx")
        drop_list = drop_list_sarc + drop_list_time
        X = df.drop(drop_list, axis=1).to_numpy()
        y = df["sarc"].to_numpy()
        target = "sarc"
        feature_list = [x for x in df.columns if x not in drop_list]
        feature_list = np.array(feature_list)

    elif dataset == 'noncon_fgrade':
        # df = pd.read_excel(path+"temp_noncon-healthmyne-clinicalMLanon_imp.xlsx")
        df = pd.read_excel(path + "temp_noncon-healthmyne-clinicalMLanon.xlsx")
        drop_list = drop_list_fgrade + drop_list_time
        X = df.drop(drop_list, axis=1).to_numpy()
        y = df["fgrade"].to_numpy()
        target = "fgrade"
        feature_list = [x for x in df.columns if x not in drop_list]
        feature_list = np.array(feature_list)

    elif dataset == 'pv_fgrade':
        # df = pd.read_excel(path+"temp_pv-healthmyne-clinicalanon_imp.xlsx")
        df = pd.read_excel(path + "temp_pv-healthmyne-clinicalanon.xlsx")
        X = df.drop(drop_list_fgrade, axis=1).to_numpy()
        y = df["fgrade"].to_numpy()
        target = "fgrade"
        feature_list = [x for x in df.columns if x not in drop_list_fgrade]
        feature_list = np.array(feature_list)

    # X, feature_list = remove_features_with_nan(X, feature_list)
    print("Shape of X is :{}".format(X.shape))
    print("Length of feature is : {}".format(len(feature_list)))

    return X, y, feature_list, df, target


def runRFE(dataset, type, clfname='rf'):
    print("\n\nCurrent dataset is " + dataset + "\n")
    path = "Dataset/Original files/"
    # Choose the dataset accordingly and set the X,y and feature values
    X, y, feature_list, df, target = get_dataset(dataset=dataset, path=path)

    # Split the train and test dataset.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Some diagnostics to get an idea about the dataset
    print("-------Diagnostics-------")
    print("The shape of X_train, y_train are {}, {}".format(X_train.shape, y_train.shape))
    print("The shape of X_test, y_test are {}, {}\n".format(X_test.shape, y_test.shape))

    '''
    In the below mentioned cases, 'rfecv', 'rfe' and 'elim_rfe' were only created for testing purpose. 
    You may not use this and can only focus on the cases of 'elim_rfecv' and 'no_rfe'. The other code is 
    not relevant much but you may check to get an idea on that approach!!
    '''

    if type=='rfecv':
        # This performs RFE with CV only once and gives the best features as observed from the train dataset
        featureSelectAndClassifyRFECV(X_train, X_test, y_train, y_test, clfname)

    elif type=='rfe':
        # This performs RFE without CV only once and gives the best features as observed from the train dataset
        featureranks = featureSelectAndClassifyRFE(X, X_test, y, y_test, clfname)
        writeToExcelFile(featureranks)

    elif type=='elim_rfecv':
        # This performs RFE with CV using nested cross-validation to eliminate features in every loop. At the end of all
        # the outer loops, the final list of features is saved in a file and the dataset with reduced features is sent
        # for testing again using CV with Support Vector Machine Classifier and Random Forest Classifier
        nestedCV_temp(X, y, feature_list)
        #XNew = eliminateFeaturesRecursivelyWithCV(X, y, clfname=clfname, feature_list=feature_list)
        #print("Shape of final data input is {}".format(XNew.shape))
        #noRFE(XNew, y, 'rf')
    #
    elif type=='no_rfe':
        # This is the baseline approach to check the performs of SVMs and RF over the entire dataset. Used for comparison.
        noRFE(X, y, scale=True)

    elif type=='just_rf':
        justRF_temp(X, y, feature_list)
        #X_new = justRF(X, y, feature_list)
        #noRFE(X_new, y, 'rf')

    elif type=='pearson':
        print("\n This is baseline accuracy\n")
        noRFE(X, y, scale=True)
        permute = False
        Xp, yp, feature_list = pearson_temp(df, dataset, target)
        runPostFiltering(Xp, yp, feature_list=feature_list)
        if permute==True:
            print("\nRunning with permuted target variables\n")
            yp = np.random.permutation(yp)
            runPostFiltering(Xp, yp, feature_list=feature_list)

    elif type=='permutation_test':
        print("\n Running permutation test score\n")
        permutationTest(X, y, feature_list, dataset)

    elif type=='runOriginal':
        X_new = remove_features_with_nan(X)


    elif type=='pca':
        PCAFilter(dataset, feature_list, X, y)

    elif type=='just_xgb':
        justXgBoost(X, y)

    else:
        print("Error!!! Incorrect type of RFE selected!! Please check type argument again.")

    return


def remove_features_with_nan(X, feature_list):
    print(X.shape)
    indices = []
    list_indices = []
    for i in range(X.shape[1]):
        col = X[:,i]
        if np.isnan(np.sum(col))==True:
            indices.append(i)
        else:
            list_indices.append(i)

    X_final = np.delete(arr=X, obj=indices, axis=1)
    feature_list = feature_list[list_indices]

    return X_final, feature_list


def rfecv_xgboost(dataset):
    path = "Dataset/Original files/"
    print("\n Dataset is: {}\n".format(dataset))
    X, y, feature_list, _, _ = get_dataset(dataset=dataset, path=path)
    estimator = xgb.XGBClassifier()
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    rfecv = RFECV(estimator=clone(estimator), step=1, cv=cv, scoring='accuracy')
    rfecv.fit(X, y)
    print("Optimal number of features is {}".format(rfecv.n_features_))
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()


def feature_selection_only(dataset, inputs):
    X, y, feature_list, df, _ = get_dataset(dataset=dataset)
    model = inputs['model']
    _ = feature_assessment_and_selection(dataset=dataset, model=model, X=X, y=y,
                                         df=df, feature_list=feature_list)

    return


def run_model(dataset, inputs):
    X, y, feature_list, df, _ = get_dataset(dataset=dataset, path=path)
    # pos_dist = 100 * sum(y) / y.shape[0]
    # print("For all cells, positive classes = {}, negative = {}".format(pos_dist, 100 - pos_dist))
    print("Type of classifier is " + inputs["type"])
    model = inputs["model"]
    impute = inputs["impute"]
    feature_select = inputs["feature_select"]
    nested_cross_validation(dataset=dataset, model=model, X=X, y=y, df=df, feature_list=feature_list,
                            impute=impute, feature_select=feature_select)

    return


def hold_code():
    optimal_features = len(best_list)
    print("Length of feature list is {} and scores is {}".format(len(list1), len(scores)))
    print("best score is {} for {} number of features".format(best_score, optimal_features))
    print("List of best features is {}".format(best_list))
    X_final = df[best_list].to_numpy()
    rs = randint(1, 100)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
    metric_scores = {
        'acc_score': mean(cross_val_score(clone(model), X_final, y, cv=cv, scoring='accuracy')),
        'f1_score': mean(cross_val_score(clone(model), X_final, y, cv=cv, scoring='f1')),
        'p_score': mean(cross_val_score(clone(model), X_final, y, cv=cv, scoring='precision')),
        'r_score': mean(cross_val_score(clone(model), X_final, y, cv=cv, scoring='recall')),
        'auc_score': mean(cross_val_score(clone(model), X_final, y, cv=cv, scoring='roc_auc'))
    }
    print(metric_scores)
    # gmean_score = mean(cross_val_score(clone(model), X_curr, y, cv=cv, scoring='f1'))
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score")
    plt.plot(optimal_features, best_score, 'ro')
    plt.annotate('(features = %f, score = %.2f)' % (optimal_features, best_score),
                 xy=(optimal_features, best_score),
                 xytext=(optimal_features + 20, best_score), arrowprops=dict(arrowstyle="->"))
    plt.plot(range(1, len(list1) + 1), scores)
    plt.title("F1 score v/s number of features for {}".format(dataset))
    # plt.savefig("Dataset/" + dataset + "_scores.jpg")
    plt.show()


def find_missing_percentage(dataset):
    path = "Dataset/Original files/"
    print("\n Dataset is: {}\n".format(dataset))
    X, _, _, _, _ = get_dataset(dataset=dataset, path=path)
    print(X.shape[0] * X.shape[1])
    new_x = np.isnan(X)
    print(new_x.sum()/(new_x.shape[0] * new_x.shape[1])*100)

    return


def main():
    inputs_xgb = {
        "type": "XGB Classifier",
        "model": xgb.XGBClassifier(),
        "impute": False,
        "feature_select": True
    }

    inputs_rf = {
        "type": "Random Forest",
        "model": RandomForestClassifier(n_estimators=100, max_depth=50),
        "impute": True,
        "feature_select": True
    }

    inputs_svm = {
        "type": "Linear SVM",
        "model": svm.LinearSVC(),
        "impute": True,
        "feature_select": False
    }

    # find_missing_percentage(dataset='noncon_sarc')
    # find_missing_percentage(dataset='pv_sarc')
    # find_missing_percentage(dataset='noncon_fgrade')
    # find_missing_percentage(dataset='pv_fgrade')

    # feature_selection_only(dataset='noncon_sarc', inputs=inputs_xgb)
    feature_selection_only(dataset='pv_sarc', inputs=inputs_xgb)
    # feature_selection_only(dataset='noncon_fgrade', inputs=inputs_xgb)
    # feature_selection_only(dataset='pv_fgrade', inputs=inputs_xgb)

    # runModel(dataset='noncon_sarc', inputs=inputs_xgb)
    # runModel(dataset='pv_sarc', inputs=inputs_xgb)
    # runModel(dataset='noncon_fgrade', inputs=inputs_xgb)
    # runModel(dataset='pv_fgrade', inputs=inputs_xgb)

    # runModel(dataset='noncon_sarc', inputs=inputs_rf)
    # runModel(dataset='pv_sarc', inputs=inputs_rf)
    # runModel(dataset='noncon_fgrade', inputs=inputs_rf)
    # runModel(dataset='pv_fgrade', inputs=inputs_rf)

    # rfecv_xgboost(dataset='noncon_sarc')
    # rfecv_xgboost(dataset='pv_sarc')
    # rfecv_xgboost(dataset='noncon_fgrade')
    # rfecv_xgboost(dataset='pv_fgrade')

    # runRFE(dataset='noncon_sarc', type='permutation_test')
    # runRFE(dataset='pv_sarc', type='permutation_test')
    # runRFE(dataset='noncon_fgrade', type='permutation_test')
    # runRFE(dataset='pv_fgrade', type='permutation_test')

    return


main()