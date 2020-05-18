from sklearn.model_selection import train_test_split
from featureSelectAndClassify import *
import pandas as pd
import numpy as np

def main(dataset='pv', type=None, clfname='svc'):
    if dataset=='noncon':
        df = pd.read_excel("Dataset/temp_noncon_imp.xlsx")
    elif dataset=='pv':
        df = pd.read_excel("Dataset/temp_pv_imp.xlsx")
    data = df.values
    feature_list = df.columns[2:]
    X = data[:, 2:]
    y = data[:, 1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print("-------Diagnostics-------")
    print("The shape of X_train, y_train are {}, {}".format(X_train.shape, y_train.shape))
    print("The shape of X_test, y_test are {}, {}\n".format(X_test.shape, y_test.shape))

    if type=='rfecv':
        featureSelectAndClassifyRFECV(X_train, X_test, y_train, y_test, clfname)
    elif type=='rfe':
        featureranks = featureSelectAndClassifyRFE(X, X_test, y, y_test, clfname)
        writeToExcelFile(featureranks)
    elif type=='elim_rfe':
        X_trainNew, X_testNew = eliminateFeaturesRecursively(dataset, X_train, X_test, y_train, y_test, feature_list, clfname=clfname)
        XNew = np.vstack((X_trainNew, X_testNew))
        #yNew = np.hstack((y_train, y_test)).T
        print("Shape of final data input is {}".format(XNew.shape))
        noRFE(XNew, y, 'svc')
        noRFE(XNew, y, 'rf')
    else:
        noRFE(X, y, clfname=clfname, scale=True)

main('pv', 'elim_rfe', clfname='svc')