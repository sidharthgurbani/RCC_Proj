from sklearn.model_selection import train_test_split
from featureSelectAndClassify import featureSelectAndClassifyRFE, featureSelectAndClassifyRFECV, writeToExcelFile
from eliminateFeatures import *
import warnings
import pandas as pd
import numpy as np
import models

warnings.filterwarnings("ignore")
def getDataset(dataset):
    # Choose the dataset accordingly and set the X,y and feature values
    if dataset == 'noncon_sarc':
        df = pd.read_excel("Dataset/temp_noncon_imp.xlsx")
        X = df.drop(["Case", "sarc"], axis=1)
        y = df["sarc"]
        target = "sarc"
        feature_list = df.columns[2:]

    elif dataset == 'pv_sarc':
        df = pd.read_excel("Dataset/temp_pv_imp.xlsx")
        X = df.drop(["Case", "sarc"], axis=1)
        y = df["sarc"]
        target = "sarc"
        feature_list = df.columns[2:]

    elif dataset == 'noncon_fgrade':
        df = pd.read_excel("Dataset/temp_noncon-healthmyne-clinicalMLanon_imp.xlsx")
        X = df.drop(["Case", "rcctype", "fgrade", "perineph", "recurr", "fucond", "deceased", "survival"], axis=1)
        y = df["fgrade"]
        target = "fgrade"
        feature_list = df.columns[8:]

    elif dataset == 'pv_fgrade':
        df = pd.read_excel("Dataset/temp_pv-healthmyne-clinicalanon_imp.xlsx")
        X = df.drop(["Case", "rcctype", "fgrade", "perineph", "recurr", "fucond", "deceased", "survival"], axis=1)
        y = df["fgrade"]
        target = "fgrade"
        feature_list = df.columns[8:]

    return X, y, feature_list, df, target


def runRFE(dataset, type, clfname):
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
        # print(X.shape)
        # y = np.resize(y, (y.shape[0], 1))
        # print(y.shape)
        # data = np.hstack((y, X))
        #Xp, yp, feature_list = pearson_temp(df, dataset, target)
        #print("Shape of X after filtering is {}".format(Xp.shape))
        # cor_matr = np.corrcoef(x=data, rowvar=False)
        # print(cor_matr.shape)
        # pdf = pd.DataFrame(data=cor_matr)
        # pdf.to_excel("corr_matrix.xlsx")
        permutationTest(X, y, feature_list, dataset)


    else:
        print("Error!!! Incorrect type of RFE selected!! Please check type argument again.")


def main():
    runRFE(dataset='noncon_sarc', type='just_rf', clfname='rf')
    # runRFE(dataset='pv_sarc', type='permutation_test', clfname='rf')
    # runRFE(dataset='noncon_fgrade', type='permutation_test', clfname='rf')
    # runRFE(dataset='pv_fgrade', type='permutation_test', clfname='rf')

main()