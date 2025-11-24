import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
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

def basic_cleaning(df):
    # Filter by year
    if YEAR_COL in df.columns:
        df = df[df[YEAR_COL].isin(FILTER_YEARS)]
        print(f"\n➡️ Filtered to years {FILTER_YEARS}. New shape: {df.shape}")

    # Fill numeric / categorical missing values
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if TARGET_COL in num_cols: num_cols.remove(TARGET_COL)
    if TARGET_COL in cat_cols: cat_cols.remove(TARGET_COL)

    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0])

    # Map “AddressChange_Claim”
    if "AddressChange_Claim" in df.columns:
        df["AddressChange_Claim"] = df["AddressChange_Claim"].astype(str).map(
            lambda x: 0 if "no change" in x.lower() else 1
        )

    # Ensure binary target
    df[TARGET_COL] = df[TARGET_COL].map(lambda x: 1 if int(x) == 1 else 0)

    return df

def prepare_features(df):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Factorize categorical
    cat_cols = X.select_dtypes(include=["object"]).columns
    for c in cat_cols:
        X[c] = pd.factorize(X[c])[0]

    # Make numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    return X, y

def handle_imbalance_with_smote(X_train, y_train):
    try:
        from imblearn.over_sampling import SMOTE
        print("\n➡️ Using SMOTE for oversampling (class imbalance fix)")
        sm = SMOTE(random_state=RANDOM_STATE)
        return sm.fit_resample(X_train, y_train), True
    except:
        print("\n⚠️ SMOTE not available — using scale_pos_weight instead")
        return (X_train, y_train), False

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
    gs = GridSearchCV(model, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)

    print("\n🎯 Best Parameters:", gs.best_params_)
    print("📈 Best CV ROC-AUC:", gs.best_score_)

    return gs.best_estimator_

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n==================== MODEL PERFORMANCE ====================")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def main():
    df = load_data(DATA_PATH)
    print("Loaded data shape:", df.shape)

    df = basic_cleaning(df)
    X, y = prepare_features(df)

    print("\nClass distribution:", y.value_counts().to_dict())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
    )

    # Handle imbalance
    (X_train, y_train), used_smote = handle_imbalance_with_smote(X_train, y_train)

    # If SMOTE not used → compute scale_pos_weight
    if not used_smote:
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        scale_pos_weight = neg / pos
    else:
        scale_pos_weight = 1

    print(f"\n➡️ scale_pos_weight used: {scale_pos_weight}")

    # Train
    best_model = train_and_tune(X_train, y_train, scale_pos_weight)

    # Evaluate
    evaluate(best_model, X_test, y_test)

if __name__ == "__main__":
    main()
