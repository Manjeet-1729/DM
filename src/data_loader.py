"""
data_loader.py
---------------
Loads the raw Dry Bean Dataset (UCI ML Repository, ID 602) and logs
everything you'll want for the "Explain the dataset" slide:
  - source
  - shape
  - class distribution (checks for imbalance)
  - missing value count
  - basic dtype summary
"""

import os

import pandas as pd

RAW_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "Dry_Bean_Dataset.xlsx"
)

DATA_SOURCE = (
    "UCI Machine Learning Repository - Dry Bean Dataset (ID 602)\n"
    "https://archive.ics.uci.edu/dataset/602/dry+bean+dataset\n"
    "Koklu, M. and Ozkan, I.A. (2020), Multiclass Classification of Dry Beans Using "
    "Computer Vision and Machine Learning Techniques. Computers and Electronics in "
    "Agriculture, 174, 105507."
)


def load_raw_data(logger, path: str = RAW_DATA_PATH) -> pd.DataFrame:
    logger.info("=" * 70)
    logger.info("STAGE 1: DATA LOADING")
    logger.info("=" * 70)
    logger.info(f"Dataset source:\n{DATA_SOURCE}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find dataset at {path}. Download Dry_Bean_Dataset.xlsx from the "
            f"UCI repository above and place it in data/raw/."
        )

    df = pd.read_excel(path)
    logger.info(f"Loaded raw data from: {path}")
    logger.info(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    logger.info(f"Columns: {list(df.columns)}")

    missing = df.isnull().sum().sum()
    logger.info(f"Total missing values: {missing}")
    if missing == 0:
        logger.info("No missing-value imputation required (confirmed no nulls in raw data).")

    dup = df.duplicated().sum()
    logger.info(f"Duplicate rows: {dup}")

    logger.info("Class distribution (target = 'Class'):")
    class_counts = df["Class"].value_counts()
    for cls, count in class_counts.items():
        pct = 100 * count / len(df)
        logger.info(f"  {cls:10s}: {count:5d}  ({pct:5.2f}%)")

    imbalance_ratio = class_counts.max() / class_counts.min()
    logger.info(f"Class imbalance ratio (largest/smallest class): {imbalance_ratio:.2f}x")

    logger.info("Dtype summary:")
    for col, dt in df.dtypes.items():
        logger.info(f"  {col:20s}: {dt}")

    return df
