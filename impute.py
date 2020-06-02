import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def impute():
    name = "Dataset/temp_noncon-healthmyne-clinicalMLanon.xlsx"
    name1 = "Dataset/temp_noncon-healthmyne-clinicalMLanon_imp.xlsx"
    df = pd.read_excel(name)
    data = df.values
    X = data[:,8:]
    y = data[:,2]
    print(X.shape)
    print(y.shape)
    imp = IterativeImputer()
    impData = imp.fit_transform(data)
    print(impData.shape)
    outdf = pd.DataFrame(impData)
    outdf.to_excel(excel_writer=name1)