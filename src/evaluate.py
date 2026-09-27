"""
evaluate.py
-----------
Final stage: train each algorithm with (a) default/baseline
hyperparameters and (b) the best hyperparameters found during
sensitivity analysis, evaluate BOTH on the untouched test set, and
produce the plots/metrics used in the "Results", "Discussion and
Conclusion" slides.
"""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neural_network import MLPClassifier

from src.utils import METRICS_DIR, save_plot

# Published benchmark from Koklu & Ozkan (2020), the paper that introduced
# this dataset -- their best model (SVM) reached 93.13% test accuracy.
PUBLISHED_BENCHMARK = {"model": "SVM (Koklu & Ozkan, 2020, original paper)", "accuracy": 0.9313}


def _evaluate_model(model, X_test, y_test, class_names, name, logger):
    t0 = time.time()
    y_pred = model.predict(X_test)
    infer_time = time.time() - t0

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    logger.info(f"[{name}] Test Accuracy : {acc:.4f}")
    logger.info(f"[{name}] Macro Precision: {prec:.4f}")
    logger.info(f"[{name}] Macro Recall   : {rec:.4f}")
    logger.info(f"[{name}] Macro F1-score : {f1:.4f}")
    logger.info(f"[{name}] Inference time on test set: {infer_time:.3f}s")
    logger.info(f"[{name}] Full classification report:\n" +
                classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues", colorbar=False)
    plt.xticks(rotation=45, ha="right")
    ax.set_title(f"Confusion Matrix -- {name}")
    save_plot(fig, f"confusion_matrix_{name.replace(' ', '_')}.png", logger)

    return {
        "model": name,
        "accuracy": acc,
        "macro_precision": prec,
        "macro_recall": rec,
        "macro_f1": f1,
        "inference_time_s": infer_time,
    }


def run_final_evaluation(data, mlp_best_params, rf_best_params, logger):
    logger.info("=" * 70)
    logger.info("STAGE 4: FINAL TRAINING + TEST-SET EVALUATION")
    logger.info("=" * 70)

    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    class_names = data["class_names"]

    all_results = []

    # ---- Baselines (default hyperparameters) ----
    logger.info("--- Baseline models (library default hyperparameters) ---")
    mlp_base = MLPClassifier(max_iter=300, early_stopping=True, random_state=42)
    mlp_base.fit(X_train, y_train)
    all_results.append(_evaluate_model(mlp_base, X_test, y_test, class_names, "MLP (baseline)", logger))

    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_base.fit(X_train, y_train)
    all_results.append(_evaluate_model(rf_base, X_test, y_test, class_names, "RandomForest (baseline)", logger))

    # ---- Tuned models (best params from sensitivity analysis) ----
    logger.info(f"--- Tuned MLP: {mlp_best_params} ---")
    mlp_tuned = MLPClassifier(max_iter=300, early_stopping=True, random_state=42, **mlp_best_params)
    mlp_tuned.fit(X_train, y_train)
    mlp_result = _evaluate_model(mlp_tuned, X_test, y_test, class_names, "MLP (tuned)", logger)
    all_results.append(mlp_result)

    logger.info(f"--- Tuned RandomForest: {rf_best_params} ---")
    rf_tuned = RandomForestClassifier(random_state=42, n_jobs=-1, **rf_best_params)
    rf_tuned.fit(X_train, y_train)
    rf_result = _evaluate_model(rf_tuned, X_test, y_test, class_names, "RandomForest (tuned)", logger)
    all_results.append(rf_result)

    # ---- Compare against published benchmark ----
    logger.info("--- Comparison against published benchmark ---")
    logger.info(
        f"Published: {PUBLISHED_BENCHMARK['model']} = {PUBLISHED_BENCHMARK['accuracy']:.4f}"
    )
    for r in [mlp_result, rf_result]:
        delta = r["accuracy"] - PUBLISHED_BENCHMARK["accuracy"]
        sign = "+" if delta >= 0 else ""
        logger.info(f"  {r['model']}: {r['accuracy']:.4f}  ({sign}{delta*100:.2f} pts vs published SVM)")

    # ---- Random Forest feature importance (bonus insight for discussion) ----
    importances = pd.Series(rf_tuned.feature_importances_, index=data["feature_names"]).sort_values(
        ascending=False
    )
    logger.info("Random Forest feature importances (top 5):")
    for feat, imp in importances.head(5).items():
        logger.info(f"  {feat:20s}: {imp:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    importances.head(10).sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title("Random Forest -- Top 10 Feature Importances")
    ax.set_xlabel("Importance")
    save_plot(fig, "rf_feature_importance.png", logger)

    # ---- Summary comparison bar chart ----
    results_df = pd.DataFrame(all_results)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(results_df["model"], results_df["accuracy"], color=["#B0B0B0", "#B0B0B0", "#4C72B0", "#DD8452"])
    ax.axhline(PUBLISHED_BENCHMARK["accuracy"], color="red", linestyle="--",
               label=f"Published benchmark ({PUBLISHED_BENCHMARK['accuracy']:.4f})")
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Model Comparison: Baseline vs Tuned vs Published Benchmark")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.4f}", (bar.get_x() + bar.get_width() / 2, height),
                    ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=15, ha="right")
    ax.legend()
    save_plot(fig, "final_comparison.png", logger)

    # ---- Save metrics to disk ----
    results_df.to_csv(os.path.join(METRICS_DIR, "final_results.csv"), index=False)
    with open(os.path.join(METRICS_DIR, "best_params.json"), "w") as f:
        json.dump({"mlp_best_params": mlp_best_params, "rf_best_params": rf_best_params,
                    "published_benchmark": PUBLISHED_BENCHMARK}, f, indent=2)
    logger.info(f"Saved final metrics table and best params to {METRICS_DIR}")

    return results_df
