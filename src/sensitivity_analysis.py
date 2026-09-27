"""
sensitivity_analysis.py
------------------------
This is the heart of the "sensitivity analysis" requirement: for each
algorithm, we vary its key hyperparameters one experiment at a time,
measure cross-validated accuracy on the TRAINING data only (the test
set stays untouched until the very end), and log every single run.

Algorithm 1: MLPClassifier (Neural Network)
    - varied: hidden layer architecture, activation function, learning rate

Algorithm 2: RandomForestClassifier
    - varied: number of trees, max depth, max features per split

Everything logged here is exactly what your "Detail results from
different algorithms" slide needs.
"""

import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

CV_FOLDS = 5


# ----------------------------------------------------------------------
# MLP sensitivity analysis
# ----------------------------------------------------------------------
MLP_GRID = [
    # (hidden_layer_sizes, activation, learning_rate_init)
    ((16,), "relu", 0.001),
    ((32,), "relu", 0.001),
    ((64,), "relu", 0.001),
    ((64, 32), "relu", 0.001),
    ((100, 50), "relu", 0.001),
    ((64, 32), "tanh", 0.001),
    ((64, 32), "logistic", 0.001),
    ((64, 32), "relu", 0.01),
    ((64, 32), "relu", 0.0001),
    ((100, 50, 25), "relu", 0.001),
]


def mlp_sensitivity_analysis(X_train, y_train, logger):
    logger.info("=" * 70)
    logger.info("STAGE 3a: SENSITIVITY ANALYSIS -- MLP (Neural Network)")
    logger.info("=" * 70)
    logger.info(f"Using {CV_FOLDS}-fold cross-validation on the TRAINING set only.")
    logger.info("Varying: hidden_layer_sizes, activation function, learning_rate_init")

    records = []
    for hidden_layers, activation, lr in MLP_GRID:
        t0 = time.time()
        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            learning_rate_init=lr,
            max_iter=300,
            early_stopping=True,
            random_state=42,
        )
        scores = cross_val_score(clf, X_train, y_train, cv=CV_FOLDS, scoring="accuracy", n_jobs=-1)
        mean_acc, std_acc = scores.mean(), scores.std()
        elapsed = time.time() - t0

        config_str = f"layers={hidden_layers}, activation={activation}, lr={lr}"
        logger.info(
            f"  [MLP] {config_str:55s} -> CV accuracy = {mean_acc:.4f} (+/-{std_acc:.4f})  "
            f"[{elapsed:.1f}s]"
        )

        records.append({
            "algorithm": "MLP",
            "config": config_str,
            "hidden_layer_sizes": str(hidden_layers),
            "activation": activation,
            "learning_rate_init": lr,
            "mean_cv_accuracy": mean_acc,
            "std_cv_accuracy": std_acc,
            "train_time_s": elapsed,
        })

    results_df = pd.DataFrame(records)
    best_row = results_df.loc[results_df["mean_cv_accuracy"].idxmax()]
    best_params = {
        "hidden_layer_sizes": eval(best_row["hidden_layer_sizes"]),
        "activation": best_row["activation"],
        "learning_rate_init": best_row["learning_rate_init"],
    }
    logger.info(f"BEST MLP CONFIG: {best_row['config']} -> {best_row['mean_cv_accuracy']:.4f}")

    return results_df, best_params


# ----------------------------------------------------------------------
# Random Forest sensitivity analysis
# ----------------------------------------------------------------------
RF_GRID = [
    # (n_estimators, max_depth, max_features)
    (50, None, "sqrt"),
    (100, None, "sqrt"),
    (200, None, "sqrt"),
    (300, None, "sqrt"),
    (100, 5, "sqrt"),
    (100, 10, "sqrt"),
    (100, 20, "sqrt"),
    (100, None, "log2"),
    (100, None, None),  # use all features per split
    (200, 15, "sqrt"),
]


def rf_sensitivity_analysis(X_train, y_train, logger):
    logger.info("=" * 70)
    logger.info("STAGE 3b: SENSITIVITY ANALYSIS -- Random Forest")
    logger.info("=" * 70)
    logger.info(f"Using {CV_FOLDS}-fold cross-validation on the TRAINING set only.")
    logger.info("Varying: n_estimators, max_depth, max_features")

    records = []
    for n_est, max_depth, max_feat in RF_GRID:
        t0 = time.time()
        clf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=max_depth,
            max_features=max_feat,
            random_state=42,
            n_jobs=-1,
        )
        scores = cross_val_score(clf, X_train, y_train, cv=CV_FOLDS, scoring="accuracy", n_jobs=-1)
        mean_acc, std_acc = scores.mean(), scores.std()
        elapsed = time.time() - t0

        config_str = f"n_estimators={n_est}, max_depth={max_depth}, max_features={max_feat}"
        logger.info(
            f"  [RF]  {config_str:55s} -> CV accuracy = {mean_acc:.4f} (+/-{std_acc:.4f})  "
            f"[{elapsed:.1f}s]"
        )

        records.append({
            "algorithm": "RandomForest",
            "config": config_str,
            "n_estimators": n_est,
            "max_depth": str(max_depth),
            "max_features": str(max_feat),
            "mean_cv_accuracy": mean_acc,
            "std_cv_accuracy": std_acc,
            "train_time_s": elapsed,
        })

    results_df = pd.DataFrame(records)
    best_row = results_df.loc[results_df["mean_cv_accuracy"].idxmax()]
    best_params = {
        "n_estimators": int(best_row["n_estimators"]),
        "max_depth": None if best_row["max_depth"] == "None" else int(best_row["max_depth"]),
        "max_features": None if best_row["max_features"] == "None" else best_row["max_features"],
    }
    logger.info(f"BEST RF CONFIG: {best_row['config']} -> {best_row['mean_cv_accuracy']:.4f}")

    return results_df, best_params
