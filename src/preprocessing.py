"""
preprocessing.py
-----------------
Dataset-handling stage: label encoding of the target, stratified
train/test split, and feature scaling (required for the MLP, harmless
for the Random Forest). Every decision is logged so it can be quoted
directly on the "Dataset Handling" slide.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROCESSED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed"
)


def preprocess(df: pd.DataFrame, logger, test_size: float = 0.2, random_state: int = 42):
    logger.info("=" * 70)
    logger.info("STAGE 2: DATASET HANDLING / PREPROCESSING")
    logger.info("=" * 70)

    X = df.drop(columns=["Class"])
    y_raw = df["Class"]

    logger.info(f"Feature matrix X: {X.shape}")
    logger.info("No missing values to impute (verified in data loading stage).")
    logger.info("No categorical feature columns present -> no one-hot encoding needed for X.")

    # Encode the target string labels -> integers (needed by sklearn estimators
    # and for consistent confusion-matrix ordering)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    logger.info(f"Label-encoded target classes: {dict(zip(le.classes_, range(len(le.classes_))))}")

    # Stratified split keeps class proportions identical in train/test,
    # important given the ~7x imbalance between BOMBAY and DERMASON
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    logger.info(
        f"Stratified train/test split ({int((1-test_size)*100)}/{int(test_size*100)}), "
        f"random_state={random_state}"
    )
    logger.info(f"Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")

    # Standardize features: zero mean, unit variance.
    # Fit ONLY on training data to avoid test-set leakage, then apply the
    # same transform to the test set.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logger.info("Applied StandardScaler (fit on train only, applied to test) -- required for MLP "
                "convergence; does not affect Random Forest's splits but is kept identical across "
                "both algorithms for a fair, reproducible comparison.")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(PROCESSED_DIR, "scaler.joblib"))
    joblib.dump(le, os.path.join(PROCESSED_DIR, "label_encoder.joblib"))
    logger.info(f"Saved fitted scaler and label encoder to {PROCESSED_DIR}")

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(X.columns),
        "class_names": list(le.classes_),
        "label_encoder": le,
        "scaler": scaler,
    }
