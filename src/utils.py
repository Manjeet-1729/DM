"""
utils.py
--------
Shared helpers: logging setup and small plotting utilities.

Every stage of the pipeline (data loading, preprocessing, sensitivity
analysis, final evaluation) writes to the SAME timestamped log file so
that the whole run can be reconstructed afterwards -- this is exactly
the log you can screenshot / paste into your presentation slides to
prove the sensitivity analysis was actually performed.
"""

import logging
import os
import sys
import time

import matplotlib.pyplot as plt

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


def get_logger(run_timestamp: str = None) -> logging.Logger:
    """
    Returns a logger that writes to both the console and a timestamped
    file under logs/. Calling this multiple times with the same
    run_timestamp reuses the same log file (all pipeline stages share
    one log per run).
    """
    if run_timestamp is None:
        run_timestamp = time.strftime("%Y%m%d_%H%M%S")

    logger = logging.getLogger("drybean")
    logger.setLevel(logging.INFO)

    # avoid duplicate handlers if get_logger() is called more than once
    if logger.handlers:
        return logger

    log_path = os.path.join(LOG_DIR, f"run_{run_timestamp}.log")

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging to {log_path}")
    return logger


def save_plot(fig, filename: str, logger=None):
    """Save a matplotlib figure to results/plots and log it."""
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if logger:
        logger.info(f"Saved plot -> {path}")
    return path
