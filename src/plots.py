"""
plots.py
--------
Visualizes the sensitivity-analysis sweeps (accuracy vs hyperparameter)
for both algorithms -- these are the charts that go on your
"Sensitivity Analysis" / "Results" slides.
"""

import matplotlib.pyplot as plt

from src.utils import save_plot


def plot_mlp_sensitivity(results_df, logger):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(results_df["config"], results_df["mean_cv_accuracy"], yerr=results_df["std_cv_accuracy"],
           color="#4C72B0", capsize=3)
    ax.set_ylabel("5-fold CV Accuracy")
    ax.set_title("MLP Sensitivity Analysis: Accuracy vs Hyperparameter Configuration")
    ax.set_ylim(results_df["mean_cv_accuracy"].min() - 0.02, results_df["mean_cv_accuracy"].max() + 0.01)
    plt.xticks(rotation=60, ha="right", fontsize=8)
    save_plot(fig, "mlp_sensitivity_analysis.png", logger)


def plot_rf_sensitivity(results_df, logger):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(results_df["config"], results_df["mean_cv_accuracy"], yerr=results_df["std_cv_accuracy"],
           color="#DD8452", capsize=3)
    ax.set_ylabel("5-fold CV Accuracy")
    ax.set_title("Random Forest Sensitivity Analysis: Accuracy vs Hyperparameter Configuration")
    ax.set_ylim(results_df["mean_cv_accuracy"].min() - 0.02, results_df["mean_cv_accuracy"].max() + 0.01)
    plt.xticks(rotation=60, ha="right", fontsize=8)
    save_plot(fig, "rf_sensitivity_analysis.png", logger)
