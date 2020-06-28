from sklearn.model_selection import train_test_split
from featureSelectAndClassify import featureSelectAndClassifyRFE, featureSelectAndClassifyRFECV, writeToExcelFile
from eliminateFeatures import *
from sklearn.preprocessing import StandardScaler
import warnings
import pandas as pd
import numpy as np
from boost import XGboost
import models
from sklearn.utils.estimator_checks import check_estimator


warnings.filterwarnings("ignore")
def getDataset(dataset):
    # Choose the dataset accordingly and set the X,y and feature values
    if dataset == 'noncon_sarc':
        df = pd.read_excel("Dataset/temp_noncon_imp.xlsx")
        X = df.drop(["Case", "sarc"], axis=1).to_numpy()
        y = df["sarc"].to_numpy()
        target = "sarc"
        feature_list = df.columns[2:]

    elif dataset == 'pv_sarc':
        df = pd.read_excel("Dataset/temp_pv_imp.xlsx")
        X = df.drop(["Case", "sarc"], axis=1).to_numpy()
        y = df["sarc"].to_numpy()
        target = "sarc"
        feature_list = df.columns[2:]

    elif dataset == 'noncon_fgrade':
        df = pd.read_excel("Dataset/temp_noncon-healthmyne-clinicalMLanon_imp.xlsx")
        X = df.drop(["Case", "rcctype", "fgrade", "perineph", "recurr", "fucond", "deceased", "survival"], axis=1).to_numpy()
        y = df["fgrade"].to_numpy()
        target = "fgrade"
        feature_list = df.columns[8:]

    elif dataset == 'pv_fgrade':
        df = pd.read_excel("Dataset/temp_pv-healthmyne-clinicalanon_imp.xlsx")
        X = df.drop(["Case", "rcctype", "fgrade", "perineph", "recurr", "fucond", "deceased", "survival"], axis=1).to_numpy()
        y = df["fgrade"].to_numpy()
        target = "fgrade"
        feature_list = df.columns[8:]

    return X, y, feature_list, df, target


def runRFE(dataset, type, clfname='rf'):
    print("\n\nCurrent dataset is " + dataset + "\n")
    # Choose the dataset accordingly and set the X,y and feature values
    X, y, feature_list, df, target = getDataset(dataset)

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

    # if type=='rfecv':
    #     # This performs RFE with CV only once and gives the best features as observed from the train dataset
    #     featureSelectAndClassifyRFECV(X_train, X_test, y_train, y_test, clfname)
    #
    # elif type=='rfe':
    #     # This performs RFE without CV only once and gives the best features as observed from the train dataset
    #     featureranks = featureSelectAndClassifyRFE(X, X_test, y, y_test, clfname)
    #     writeToExcelFile(featureranks)
    #
    if type=='elim_rfecv':
        # This performs RFE with CV using nested cross-validation to eliminate features in every loop. At the end of all
        # the outer loops, the final list of features is saved in a file and the dataset with reduced features is sent
        # for testing again using CV with Support Vector Machine Classifier and Random Forest Classifier
        nestedCV_temp(X, y, feature_list)
        #XNew = eliminateFeaturesRecursivelyWithCV(X, y, clfname=clfname, feature_list=feature_list)
        #print("Shape of final data input is {}".format(XNew.shape))
        #noRFE(XNew, y, 'rf')
    #
    # elif type=='no_rfe':
    #     # This is the baseline approach to check the performs of SVMs and RF over the entire dataset. Used for comparison.
    #     noRFE(X, y, scale=True)

    if type=='just_rf':
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
        removeFeaturesWithNan()

    elif type=='pca':
        PCAFilter(dataset, feature_list, X, y)

    elif type=='just_xgb':
        justXgBoost(X, y)

    else:
        print("Error!!! Incorrect type of RFE selected!! Please check type argument again.")

def FeatImpWithBoost(dataset):
    X, y, feature_list, _, _ = getDataset(dataset)
    # X, y, feature_list = removeFeaturesWithNan()
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    print(X.shape)
    XGboost(X, y, feature_list)

def main():
    FeatImpWithBoost(dataset='pv_sarc')
    # checkModel()
    # runRFE(dataset='noncon_sarc', type='xgboost')
    # runRFE(dataset='pv_sarc', type='elim_rfecv')
    # runRFE(dataset='noncon_fgrade', type='just_xgb')
    # runRFE(dataset='pv_fgrade', type='just_xgb')
    # for _ in range(1):
    #     # runRFE(dataset='noncon_sarc', type='xgboost')
    #     runRFE(dataset='pv_sarc', type='xgboost')
    #     # runRFE(dataset='noncon_fgrade', type='xgboost')
    #     # runRFE(dataset='pv_fgrade', type='xgboost')


    return






main()