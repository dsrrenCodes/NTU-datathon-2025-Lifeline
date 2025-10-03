# My Model

My model pipeline (best_model_pipeline) is a joblib file. Look at how_to_run_model.ipynb to see how to load it up to test the model. The functions in the file are required to load the model!

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required libraries.

```bash
pip install -r requirements.txt
```

## Usage



```python
import copy
import joblib
import numpy as np
import pandas as pd
import sklearn.preprocessing

# ----------------- IMPORTANT Functions -----------------

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

# ----------------- Example Pipeline -----------------

# Load your dataset assuming df is test dataset
df = pd.read_csv("your_data.csv")
# Load the trained model
final_model_pipeline = joblib.load("best_model_pipeline.joblib")

# Make predictions
df["pred"] = final_model_pipeline.predict(df)

# Map predictions to results
df = labels_to_results(df, label_col="pred")

# View results
print(df[["pred", "results"]].head())

