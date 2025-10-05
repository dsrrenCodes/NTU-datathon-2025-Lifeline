import pandas as pd
import numpy as np
import copy
import sklearn
import imblearn
import joblib

# CHANGE FILE PATH
explore=pd.read_excel(r'C:\Users\darre\OneDrive\Desktop\models (ntu datathon)\CTG.xls',sheet_name=1,header=1)
explore = explore.loc[:, ~explore.columns.str.contains('Unnamed:')]
complete_features=['b','e','LB','AC','FM','UC','DL','DS','DP','ASTV','MSTV','ALTV','MLTV','Width','Min','Max','Nmax','Nzeros','Mode','Mean','Median','Variance','Tendency','CLASS','NSP']

explore=explore[complete_features]
df=explore.copy()

#remove Y variables to stimulate test data
df=df.drop(columns=['NSP'])

# ----------external fuctions (ARE REQUIRED) TO LOAD MODEL------------------
 
def add_features(df):
    df = df.copy()
    df['heart_rate_range'] = df['Max'] - df['Min']
    df['ASTV_MSTV_ratio'] = df['ASTV'] / (df['MSTV'] + 1e-8)
    df['ALTV_MLTV_ratio'] = df['ALTV'] / (df['MLTV'] + 1e-8)
    df['Variability_Score'] = (df['ASTV'] + df['ALTV']) / 2
    df['AC_UC_interaction'] = df['AC'] * df['UC']
    return df

def outlier_removal(X, threshold=7):
    new_X = copy.deepcopy(X)
    new_X_scaled = sklearn.preprocessing.scale(X)
    new_X[abs(new_X_scaled) > threshold] = np.nan

    return new_X



def labels_to_results(df, label_col='predicted_label'):
    label_map = {1: 'Normal', 2: 'Suspect', 3: 'Pathologic'}
    df['results'] = df[label_col].map(label_map)
    return df


# ------------RUN MODEL-------------------

# CHANGE FILE PATH
final_model_pipeline = joblib.load(r'C:\Users\darre\OneDrive\Desktop\models (ntu datathon)\best_model_pipeline')

df['pred']=final_model_pipeline.predict(df)
df=labels_to_results(df,label_col='pred')
# pred are the [1,2,3]
#results are [Normal=1; Suspect=2; Pathologic=3]
df[['pred','results']]

print(df)