import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

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

# Replace 'no change' with 0 and any other value with 1 in the 'AddressChange_Claim' column
df['AddressChange_Claim'] = df['AddressChange_Claim'].map(lambda x: 0 if x == 'no change' else 1)

# Define X and y
X = df.drop('BasePolicy', axis=1)  
y = df['BasePolicy']

# Convert labels into binary values
y[y == 2] = 0  
y[y == 1] = 1  

# Preview the y labels
print(y.head())

# Convert all X columns to numeric (if necessary)
X = X.apply(pd.to_numeric, errors='coerce')

# Split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Define the parameter grid to search for hyperparameters
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'n_estimators': [100, 200],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'min_child_weight': [1, 5, 10],
}

# Initialize the XGBoost model
xgb_clf = XGBClassifier(objective='binary:logistic', scale_pos_weight=1)

# Set up k-fold cross-validation (StratifiedKFold ensures the class distribution is preserved)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Set up GridSearchCV with k-fold cross-validation
grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=kfold, 
                           scoring='accuracy', n_jobs=-1, verbose=1)

# Perform grid search with cross-validation
grid_search.fit(X_train, y_train)

# Print the best parameters and best score from the grid search
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy score: {grid_search.best_score_}")

# Retrieve the best model from grid search
best_model = grid_search.best_estimator_

# Make predictions with the best model on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model's performance
print('XGBoost model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))
