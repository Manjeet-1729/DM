# Presentation Notes — Slide-by-Slide (maps to Prof. Varun's requirements)

Use this as your speaker-notes outline. Numbers are pulled directly from `logs/run_*.log` and `results/metrics/`.

---

### Slide 1 — Title
Dry Bean Classification: MLP Neural Network vs Random Forest
Group members, CS660, date.

---

### Slide 2 — Introduction / Problem Statement (~1 min)
- Manually sorting bean varieties by eye is slow and inconsistent.
- Goal: automatically classify a bean grain into 1 of 7 varieties from 16 geometric/shape features extracted from its image.
- Real-world use: automated seed sorting, quality control, agricultural export grading.

---

### Slide 3 — Dataset (~1.5 min)
- UCI ML Repository, Dry Bean Dataset (ID 602), Koklu & Ozkan (2020).
- 13,611 samples, 16 numeric features, 7 classes.
- Class counts: DERMASON 3546, SIRA 2636, SEKER 2027, HOROZ 1928, CALI 1630, BARBUNYA 1322, BOMBAY 522.
- 0 missing values, 0 duplicate handling needed (state this explicitly — shows you checked).

---

### Slide 4 — Dataset Handling (~1.5 min)
- No imputation needed (no missing values).
- Target label-encoded to integers 0–6.
- Stratified 80/20 train/test split → 10,888 train / 2,723 test rows (preserves class ratios given the ~6.8× imbalance).
- StandardScaler fit on train only, applied to test (prevents data leakage) — required for MLP convergence.

---

### Slide 5 — Methodology (~2.5 min)
- **Algorithm 1: MLP (Neural Network)** — varied hidden layer architecture, activation function, learning rate.
- **Algorithm 2: Random Forest** — varied number of trees, max depth, max features per split.
- Sensitivity analysis: 5-fold cross-validation on training data for each of 10 configurations per algorithm (20 total experiments), picking the config with highest mean CV accuracy.
- Metric: Accuracy (primary), plus macro Precision/Recall/F1 to account for class imbalance.

---

### Slide 6 — Sensitivity Analysis Results (~1.5 min)
Insert: `results/plots/mlp_sensitivity_analysis.png` and `results/plots/rf_sensitivity_analysis.png`
- Best MLP: hidden_layer_sizes=(100,50), activation=relu, lr=0.001 → CV accuracy 0.9321
- Best RF: n_estimators=200, max_depth=None, max_features=sqrt → CV accuracy 0.9253
- Talking point: deeper/wider MLP layers and more trees both helped up to a point, then plateaued (diminishing returns) — classic sensitivity-analysis finding.

---

### Slide 7 — Final Results on Test Set (~1.5 min)
Insert: `results/plots/final_comparison.png`

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| MLP (tuned) | 0.9236 | 0.9353 |
| Random Forest (tuned) | 0.9199 | 0.9321 |
| Published SVM benchmark | 0.9313 | — |

Insert one confusion matrix per algorithm (`confusion_matrix_MLP_(tuned).png`, `confusion_matrix_RandomForest_(tuned).png`).

---

### Slide 8 — Discussion & Conclusion (~1.5 min)
- Both models land within ~1 point of the published SVM benchmark — strong result.
- MLP edged out Random Forest — continuous, scaled features suit its smooth decision boundaries.
- Tuned models scored marginally lower on test than library-default baselines, despite higher CV accuracy — a real, worth-discussing example of mild overfitting to CV folds / the value of a true held-out test set.
- Random Forest advantages: interpretable (feature importances — ShapeFactor3, Perimeter, Compactness dominate), no scaling required.
- MLP advantages: slightly higher accuracy, but needs scaling and more careful tuning to avoid overfitting.
- Both models confuse SIRA and DERMASON most often — these varieties are the most visually/morphologically similar.
- Future work: try SVM (to directly reproduce the published benchmark), or gradient boosting (XGBoost/LightGBM).

---

### Appendix (optional backup slide)
- Full run log available in `logs/run_*.log` — screenshot a snippet if the instructor wants proof of the sensitivity-analysis process.
