from sklearn.model_selection import train_test_split
from featureSelectAndClassify import featureSelectAndClassifyRFE, featureSelectAndClassifyRFECV, writeToExcelFile
from eliminateFeatures import eliminateFeaturesRecursively, eliminateFeaturesRecursivelyWithCV, noRFE, justRF
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

def runRFE(dataset, type, clfname):
    # Choose the dataset accordingly and set the X,y and feature values
    if dataset=='noncon':
        df = pd.read_excel("Dataset/temp_noncon_imp.xlsx")
        data = df.values
        feature_list = df.columns[2:]
        X = data[:, 2:]
        y = data[:, 1]
    elif dataset=='pv':
        df = pd.read_excel("Dataset/temp_pv_imp.xlsx")
        data = df.values
        feature_list = df.columns[2:]
        X = data[:, 2:]
        y = data[:, 1]
    elif dataset == 'noncon_anon':
        df = pd.read_excel("Dataset/temp_noncon-healthmyne-clinicalMLanon_imp.xlsx")
        data = df.values
        feature_list = df.columns[8:]
        X = data[:, 8:]
        y = data[:, 2]
    elif dataset=='pv_anon':
        df = pd.read_excel("Dataset/temp_pv-healthmyne-clinicalanon_imp.xlsx")
        data = df.values
        feature_list = df.columns[8:]
        X = data[:, 8:]
        y = data[:, 2]



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
        XNew = eliminateFeaturesRecursivelyWithCV(X, y, clfname=clfname, feature_list=feature_list)
        print("Shape of final data input is {}".format(XNew.shape))
        noRFE(XNew, y, 'svc')
        noRFE(XNew, y, 'rf')

    elif type=='elim_rfe':
        # This follows the procedure of 'elim_rfecv; but hard codes the number of features to eliminate in every loop
        # rather than letting the algorithm decide. Not very robust.
        X_trainNew, X_testNew = eliminateFeaturesRecursively(dataset, X_train, X_test, y_train, y_test, feature_list, clfname=clfname)
        XNew = np.vstack((X_trainNew, X_testNew))
        print("Shape of final data input is {}".format(XNew.shape))
        noRFE(XNew, y, clfname='svc')
        noRFE(XNew, y, clfname='rf')

    elif type=='no_rfe':
        # This is the baseline approach to check the performs of SVMs and RF over the entire dataset. Used for comparison.
        noRFE(X, y, clfname=clfname, scale=True)

    elif type=='just_rf':
        X_new = justRF(X, y, feature_list)
        noRFE(X_new, y, 'svc')
        noRFE(X_new, y, 'rf')

    else:
        print("Error!!! Incorrect type of RFE selected!! Please check type argument again.")


def main():
    runRFE(dataset='noncon_anon', type='elim_rfecv', clfname='svc')

main()