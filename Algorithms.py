import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

# CONFIG
DATA_PATH = "/Users/caitlyncameron/Downloads/Algorithms/DataAlgorithm.csv"
TARGET_COL = "FraudFound_P"
YEAR_COL = "Year"
FILTER_YEARS = [1994, 1995, 1996]
USE_SMOTE = True
RANDOM_STATE = 42

def load_data(path):
    return pd.read_csv(path)

# BASIC CLEANING + ALL ADVANCED FRAUD FEATURES
def basic_cleaning(df):

    # Filter by selected years

    if YEAR_COL in df.columns:
        df = df[df[YEAR_COL].isin(FILTER_YEARS)]
        print(f"\nFiltered to years {FILTER_YEARS}. New shape: {df.shape}")

    # Fill missing values
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if TARGET_COL in num_cols: num_cols.remove(TARGET_COL)
    if TARGET_COL in cat_cols: cat_cols.remove(TARGET_COL)

    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0])

    # Address Change
    if "AddressChange_Claim" in df.columns:
        df["AddressChange_Claim"] = df["AddressChange_Claim"].astype(str).map(
            lambda x: 0 if "no change" in x.lower() else 1
        )

    # FEATURE: DelayToFileClaim
    if {"Month", "MonthClaimed"}.issubset(df.columns):
        month_map = {
            "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
            "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
        }

        df["Month_num"] = df["Month"].map(month_map)
        df["MonthClaimed_num"] = df["MonthClaimed"].map(month_map)

        df["DelayToFileClaim"] = df["MonthClaimed_num"] - df["Month_num"]
        df["DelayToFileClaim"] = df["DelayToFileClaim"].apply(
            lambda x: x if x >= 0 else x + 12
        )

        print("Added DelayToFileClaim column.")

    # FEATURE: Policy Timing
    if {"Days_Policy_Accident", "Days_Policy_Claim"}.issubset(df.columns):

        df["Days_Policy_Accident"] = pd.to_numeric(df["Days_Policy_Accident"], errors="coerce")
        df["Days_Policy_Claim"] = pd.to_numeric(df["Days_Policy_Claim"], errors="coerce")

        df["Days_Policy_Accident"] = df["Days_Policy_Accident"].fillna(df["Days_Policy_Accident"].median())
        df["Days_Policy_Claim"] = df["Days_Policy_Claim"].fillna(df["Days_Policy_Claim"].median())

        df["DelayPolicyToAccident"] = df["Days_Policy_Accident"] - df["Days_Policy_Claim"]

        print("Added Days_Policy_Accident, Days_Policy_Claim, DelayPolicyToAccident.")

    # EXTRA FRAUD FEATURES -------------------------------------

    if {"TotalClaimAmount", "InsuredIncome"}.issubset(df.columns):
        df["ClaimAmountToIncomeRatio"] = df["TotalClaimAmount"] / (df["InsuredIncome"] + 1)

    if {"InsuredAge", "VehicleAge"}.issubset(df.columns):
        df["VehicleAgeDifference"] = df["InsuredAge"] - df["VehicleAge"]

    if "AccidentTime" in df.columns:
        df["AccidentTime"] = pd.to_numeric(df["AccidentTime"], errors="coerce")
        df["LateNightAccident"] = df["AccidentTime"].apply(lambda x: 1 if 2 <= x <= 5 else 0)

    if {"InsuredID", "DateOfClaim"}.issubset(df.columns):
        df["DateOfClaim"] = pd.to_datetime(df["DateOfClaim"], errors="coerce")
        df["ClaimsThisMonth"] = df.groupby("InsuredID")["DateOfClaim"].transform(
            lambda x: x.dt.to_period("M").map(x.dt.to_period("M").value_counts())
        )

    if "Days_Policy_Accident" in df.columns:
        df["AccidentSoonAfterPolicyStart"] = (df["Days_Policy_Accident"] < 14).astype(int)

    if "PoliceReportAvailable" in df.columns:
        df["MissingPoliceReport"] = df["PoliceReportAvailable"].apply(
            lambda x: 1 if str(x).upper() == "NO" else 0
        )

    print("Added all advanced fraud-indicator features.")

    df[TARGET_COL] = df[TARGET_COL].map(lambda x: 1 if int(x) == 1 else 0)

    return df

# FEATURE PREPARATION
def prepare_features(df):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    cat_cols = X.select_dtypes(include=["object"]).columns
    for c in cat_cols:
        X[c] = pd.factorize(X[c])[0]

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    return X, y

# SMOTE HANDLING
def handle_imbalance_with_smote(X_train, y_train):
    try:
        from imblearn.over_sampling import SMOTE
        print("\nUsing SMOTE for oversampling")
        sm = SMOTE(random_state=RANDOM_STATE)
        return sm.fit_resample(X_train, y_train), True
    except:
        print("\nSMOTE not available — using scale_pos_weight instead")
        return (X_train, y_train), False

# XGBOOST TRAINING + GRID SEARCH
def train_and_tune(X_train, y_train, scale_pos_weight):

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
    )

    param_grid = {
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
        "n_estimators": [100, 200],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    
    # ⬇ ROC-AUC removed. Now using F1.
    gs = GridSearchCV(model, param_grid, scoring="f1", cv=cv, n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)

    print("\nBest Parameters:", gs.best_params_)
    print("Best CV F1:", gs.best_score_)

    return gs.best_estimator_

# MODEL EVALUATION
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("\n=== MODEL PERFORMANCE ====")
    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# MAIN
def main():
    df = load_data(DATA_PATH)
    print("Loaded data shape:", df.shape)

    df = basic_cleaning(df)
    X, y = prepare_features(df)

    print("\nClass distribution:", y.value_counts().to_dict())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )

    (X_train, y_train), used_smote = handle_imbalance_with_smote(X_train, y_train)

    if not used_smote:
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        scale_pos_weight = neg / pos
    else:
        scale_pos_weight = 1

    print(f"\nscale_pos_weight used: {scale_pos_weight}")

    best_model = train_and_tune(X_train, y_train, scale_pos_weight)

    evaluate(best_model, X_test, y_test)

if __name__ == "__main__":
    main()
