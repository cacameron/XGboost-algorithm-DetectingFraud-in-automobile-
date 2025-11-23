import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import xgboost as xgb
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# Import dataset
data = '/Users/caitlyncameron/Downloads/Algorithms/DataAlgorithm.csv'
df = pd.read_csv(data)

# Check the shape and first few rows
print(df.shape)
print(df.head())

# Convert 'BasePolicy' to numeric if it's categorical
if df['BasePolicy'].dtype == 'object':
    df['BasePolicy'] = pd.factorize(df['BasePolicy'])[0]  # Convert categorical to numeric

# Define X and y
X = df.drop('BasePolicy', axis=1)  
y = df['BasePolicy']

# Convert labels into binary values
y[y == 2] = 0  # Replace all 2s with 0
y[y == 1] = 1  # Keep 1s as 1

# Preview the y labels
print(y.head())

# Convert all X columns to numeric (if necessary)
X = X.apply(pd.to_numeric, errors='coerce')

# Define data_dmatrix for XGBoost
data_dmatrix = xgb.DMatrix(data=X, label=y)

# Check the shape of the data_dmatrix
print(data_dmatrix)

# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# declare parameters
params = {
            'objective':'binary:logistic',
            'max_depth': 4,
            'alpha': 10,
            'learning_rate': 1.0,
            'n_estimators':100
        }
# instantiate the classifier 
xgb_clf = XGBClassifier(**params)



# fit the classifier to the training data
xgb_clf.fit(X_train, y_train)
# alternatively view the parameters of the xgb trained model
print(xgb_clf)

# make predictions on test data
y_pred = xgb_clf.predict(X_test)

# check accuracy score
from sklearn.metrics import accuracy_score

print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))