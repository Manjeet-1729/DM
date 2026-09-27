"""
main.py
-------
Entry point for the full CS660 project pipeline:

    1. Load raw data                      (src/data_loader.py)
    2. Preprocess (split + scale)          (src/preprocessing.py)
    3. Sensitivity analysis (MLP + RF)     (src/sensitivity_analysis.py)
    4. Final train + test evaluation       (src/evaluate.py)
    5. Plots + logs saved for the slides   (results/, logs/)

Run with:
    python main.py

Everything prints to the console AND is saved to logs/run_<timestamp>.log
so you can attach the exact log file as evidence of the sensitivity
analysis in your presentation appendix.
"""

import time

from src.data_loader import load_raw_data
from src.evaluate import run_final_evaluation
from src.plots import plot_mlp_sensitivity, plot_rf_sensitivity
from src.preprocessing import preprocess
from src.sensitivity_analysis import mlp_sensitivity_analysis, rf_sensitivity_analysis
from src.utils import METRICS_DIR, get_logger


def main():
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    logger = get_logger(run_ts)
    t_start = time.time()

    logger.info("#" * 70)
    logger.info("CS660 - Data Mining for Decision Making | Dry Bean Classification")
    logger.info("Algorithms compared: MLP Neural Network vs Random Forest")
    logger.info("#" * 70)

    # 1. Load
    df = load_raw_data(logger)

    # 2. Preprocess
    data = preprocess(df, logger)

    # 3. Sensitivity analysis
    mlp_results_df, mlp_best_params = mlp_sensitivity_analysis(data["X_train"], data["y_train"], logger)
    rf_results_df, rf_best_params = rf_sensitivity_analysis(data["X_train"], data["y_train"], logger)

    mlp_results_df.to_csv(f"{METRICS_DIR}/mlp_sensitivity_results.csv", index=False)
    rf_results_df.to_csv(f"{METRICS_DIR}/rf_sensitivity_results.csv", index=False)

    plot_mlp_sensitivity(mlp_results_df, logger)
    plot_rf_sensitivity(rf_results_df, logger)

    # 4. Final evaluation on the untouched test set
    final_results_df = run_final_evaluation(data, mlp_best_params, rf_best_params, logger)

    elapsed = time.time() - t_start
    logger.info("#" * 70)
    logger.info(f"PIPELINE COMPLETE in {elapsed/60:.1f} minutes")
    logger.info("Final results:")
    logger.info("\n" + final_results_df.to_string(index=False))
    logger.info("#" * 70)


if __name__ == "__main__":
    main()
