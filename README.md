# Dry Bean Classification — MLP vs Random Forest

**Course:** CS 660 – Data Mining for Decision Making
**Task:** Multi-class classification of 7 dry bean varieties using two ML algorithms, with sensitivity analysis on hyperparameters.

## 1. Problem Statement

Given 16 morphological features extracted from images of dry bean grains (area, perimeter, axis lengths, shape factors, etc.), predict which of 7 registered bean varieties (SEKER, BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SIRA) each grain belongs to. Automating this replaces slow, error-prone manual sorting in agricultural quality control.

## 2. Dataset

- **Source:** [UCI Machine Learning Repository — Dry Bean Dataset (ID 602)](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
- **Citation:** Koklu, M. and Ozkan, I.A. (2020). *Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques.* Computers and Electronics in Agriculture, 174, 105507.
- **Size:** 13,611 samples × 16 features + 1 target column (`Class`)
- **Missing values:** none (verified programmatically, see logs)
- **Class balance:** imbalanced — DERMASON (3,546 samples) vs BOMBAY (522 samples), ~6.8× ratio. Handled with a **stratified** train/test split so class proportions are preserved in both sets.

## 3. Algorithms Compared

| Algorithm | Sensitivity-analysis parameters varied |
|---|---|
| **MLP Neural Network** (`sklearn.neural_network.MLPClassifier`) | hidden layer architecture (1–3 layers, 16–100 neurons/layer), activation function (relu / tanh / logistic), learning rate (0.0001 – 0.01) |
| **Random Forest** (`sklearn.ensemble.RandomForestClassifier`) | number of trees (50–300), max tree depth (5, 10, 15, 20, unlimited), features considered per split (sqrt / log2 / all) |

Both are evaluated with **5-fold cross-validation on the training set only**; the test set is untouched until the final step.

## 4. Project / File Structure

```
dry-bean-classification/
├── README.md                     <- this file
├── requirements.txt
├── .gitignore
├── main.py                       <- run this: executes the full pipeline
├── data/
│   ├── raw/
│   │   └── Dry_Bean_Dataset.xlsx <- original UCI data
│   └── processed/                <- fitted scaler + label encoder (generated)
├── src/
│   ├── data_loader.py            <- Stage 1: load + log dataset stats
│   ├── preprocessing.py          <- Stage 2: split, scale, encode + log
│   ├── sensitivity_analysis.py   <- Stage 3: hyperparameter sweeps + log
│   ├── evaluate.py               <- Stage 4: final train/test + metrics + log
│   ├── plots.py                  <- sensitivity-analysis chart generation
│   └── utils.py                  <- shared logger + plot-saving helper
├── logs/
│   └── run_<timestamp>.log       <- full log of every pipeline run (generated)
└── results/
    ├── plots/                    <- all PNG charts (generated)
    └── metrics/                  <- CSV/JSON result tables (generated)
```

## 5. How to Run

```bash
git clone <your-repo-url>
cd dry-bean-classification
pip install -r requirements.txt
python main.py
```

This runs the entire pipeline (load → preprocess → sensitivity analysis → final evaluation) in one go, prints progress to the console, and writes:
- a full log file to `logs/run_<timestamp>.log`
- 8 charts to `results/plots/`
- result tables to `results/metrics/`

Total runtime: ~4 minutes on a standard CPU (no GPU needed — scikit-learn's MLP and Random Forest are CPU-only).

## 6. Results Summary

*(from the actual run — see `logs/` and `results/metrics/final_results.csv` for full detail)*

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| MLP (baseline / default params) | 0.9258 | 0.9366 |
| Random Forest (baseline / default params) | 0.9207 | 0.9333 |
| **MLP (tuned via sensitivity analysis)** | **0.9236** | 0.9353 |
| **Random Forest (tuned via sensitivity analysis)** | **0.9199** | 0.9321 |
| Published benchmark — SVM (Koklu & Ozkan, 2020) | 0.9313 | — |

**Best sensitivity-analysis configurations found (via 5-fold CV on training data):**
- MLP: `hidden_layer_sizes=(100, 50)`, `activation='relu'`, `learning_rate_init=0.001` → CV accuracy 0.9321
- Random Forest: `n_estimators=200`, `max_depth=None`, `max_features='sqrt'` → CV accuracy 0.9253

**Most important features (Random Forest):** ShapeFactor3, Perimeter, Compactness, ShapeFactor1, MinorAxisLength — the shape-descriptor features dominate over raw size features, which makes sense: bean varieties differ more in *form* than in absolute size.

## 7. Discussion (talking points for the presentation)

- Both tuned models land within ~1–1.2 percentage points of the published SVM benchmark, which is a strong result for a class project.
- Interestingly, the *tuned* models scored marginally **lower** on the held-out test set than the *default-parameter* baselines, even though tuning improved cross-validation accuracy. This is a genuine and useful discussion point: it shows the CV-selected configurations were very slightly overfit to the training folds, and that scikit-learn's sensible defaults are already close to optimal for this dataset — a good illustration of the bias–variance tradeoff and why a held-out test set (not just CV) matters.
- MLP slightly outperformed Random Forest overall, plausibly because the 16 features are continuous and reasonably smooth/linearly-separable per class after scaling, which favors a neural network's ability to learn smooth decision boundaries, whereas Random Forest's axis-aligned splits are slightly less efficient on this kind of data.
- Random Forest is much more interpretable (feature importances) and requires no feature scaling, while MLP needed standardization to converge — a real methodological tradeoff worth mentioning.
- Both models struggled most with **SIRA vs DERMASON** confusion (see confusion matrices in `results/plots/`) — these two varieties are morphologically the most similar, which matches the physical intuition about bean shapes.

## 8. Pushing to GitHub

```bash
cd dry-bean-classification
git init
git add .
git commit -m "Initial commit: Dry Bean classification project (MLP vs Random Forest)"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```
