import pandas as pd
from fancyimpute import NuclearNormMinimization

def impute(df):
    data = df.values
    X = data[:,2:]
    y = data[:,1]
    print(X.shape)
    print(y.shape)
    impData = NuclearNormMinimization().fit_transform(data)
    outdf = pd.DataFrame(impData)
    outdf.to_excel(excel_writer="../Dataset/temp_noncon_imp.xlsx")