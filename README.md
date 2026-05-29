# Titanic Survival Prediction

> Binary classification on the [Kaggle Titanic](https://www.kaggle.com/c/titanic) dataset — predicting whether a passenger survived.

---

## 1. Project Overview

This project builds an end-to-end machine-learning pipeline to predict passenger survival on the RMS Titanic. It covers data exploration, preprocessing, feature engineering, model training, hyper-parameter tuning, evaluation, and explainability — making it a complete academic case study in applied ML.

**Key highlights:**
- Three classifiers compared on a stratified hold-out set and 5-fold cross-validation
- No data leakage: all preprocessing wrapped inside `sklearn.Pipeline`
- SHAP-based global feature importance for model explainability
- Interactive Streamlit demo for live prediction

---

## 2. Dataset Description

Source: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)

| File | Rows | Description |
|------|------|-------------|
| `data/train.csv` | 891 | Labelled passengers (used for training & evaluation) |
| `data/test.csv` | 418 | Unlabelled passengers (Kaggle submission) |

**Raw features:** `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`, `Name`, `Ticket`, `Cabin`

**Engineered features (inside pipeline):**

| Feature | Description |
|---------|-------------|
| `Family_Size` | `SibSp + Parch + 1` |
| `Is_Alone` | 1 if travelling alone |
| `Title` | Extracted from passenger name (Mr, Mrs, Miss, …) |
| `Deck` | Cabin letter (proxy for ship location) |
| `Group_Size` | Number of passengers sharing the same ticket |

**Missing values:**
- `Age`: ~20% missing → imputed by median within each Title group
- `Cabin`: ~77% missing → Deck extracted where available, else "U"
- `Embarked`: 2 missing → filled with mode

---

## 3. Technologies Used

| Category | Library / Tool |
|----------|---------------|
| Language | Python 3.x |
| Data wrangling | `pandas`, `numpy` |
| ML framework | `scikit-learn` |
| Visualisation | `matplotlib` |
| Explainability | `shap` |
| Serialisation | `joblib` |
| Demo app | `streamlit` |
| Report generation | `python-docx`, `python-pptx` (optional) |

---

## 4. ML Workflow

```
Raw CSV → EDA → Cleaning → Feature Engineering
       → Encoding → Preprocessing (Pipeline)
       → Train/Test Split (stratified 80/20)
       → Model Training (LR, DT, RF)
       → GridSearchCV (DT, RF)
       → Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)
       → StratifiedKFold Cross-Validation (5-fold)
       → SHAP Explainability
       → Submission CSV + model.pkl
       → Streamlit Demo
```

> **No data leakage:** the scaler, encoder, and imputer are all fitted *after* the train/test split, inside `sklearn.Pipeline`.

---

## 5. Models Used

| Model | Type | Tuned? |
|-------|------|--------|
| Logistic Regression | Linear | No (baseline) |
| Decision Tree | Rule-based | Yes (GridSearchCV) |
| Random Forest | Ensemble / Bagging | Yes (GridSearchCV) |

**Deployment choice:** Best model by hold-out F1 → refit on full training data → `model.pkl` (currently Logistic Regression)  
**Explainability:** Random Forest + SHAP (tree-based model enables exact TreeExplainer)

---

## 6. Evaluation Metrics

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 0.8436 | — | — | **0.7879** | 0.8725 |
| Decision Tree | 0.8268 | — | — | 0.7634 | 0.8502 |
| Random Forest | 0.8101 | — | — | 0.7424 | 0.8449 |

*(Full precision/recall values shown in notebook after running all cells.)*

**Cross-validation:** 5-fold StratifiedKFold on training data, reporting mean ± std Accuracy and F1 per model.

---

## 7. Visualisations

All plots are saved to `images/` when the notebook is executed:

| File | Description |
|------|-------------|
| `survival_rate.png` | Class distribution (survived vs. not) |
| `confusion_matrix.png` | Confusion matrix for the best model |
| `roc_curve.png` | ROC curves for all three models |
| `feature_importance.png` | Random Forest feature importances (top 15) |
| `shap_summary.png` | SHAP beeswarm summary plot (Random Forest) |

---

## 8. How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Train & Evaluate

Open and run all cells in `modeling.ipynb` (e.g., with Jupyter or VS Code).  
This generates `model.pkl`, `submission.csv`, and all plots in `images/`.

### Launch Streamlit Demo

```bash
streamlit run app.py
```

### (Optional) Rebuild Word / PowerPoint Reports

```bash
pip install python-docx python-pptx
python scripts/build_reports.py
```

---

## 9. Project Structure

```
Do_An_Cuoi_Ki/
├── data/
│   ├── train.csv
│   └── test.csv
├── images/            # Auto-generated plots
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── roc_curve.png
│   ├── shap_summary.png
│   └── survival_rate.png
├── reports/
│   ├── report.docx    # Written report (Vietnamese)
│   └── slides.pptx    # Presentation deck
├── scripts/
│   └── build_reports.py
├── modeling.ipynb     # Full ML pipeline notebook
├── app.py             # Streamlit demo
├── model.pkl          # Trained pipeline (best model)
├── submission.csv     # Kaggle submission file
├── requirements.txt
└── README.md
```

---

## 10. Limitations

1. **Small dataset** — Only 891 training samples, making robust estimation difficult.
2. **Missing values** — Age (~20%) and Cabin (~77%) have significant gaps; imputation introduces assumptions.
3. **Gender bias** — The dataset strongly reflects historical social norms ("women and children first"). The model learns this pattern, which is a data bias, not a universal truth.
4. **Possible overfitting** — Despite GridSearchCV, tree-based models can overfit with small data. Cross-validation scores provide a more honest estimate than a single hold-out split.
5. **Historical data** — The model generalises only to this specific event and dataset; it cannot be applied to other contexts.

---

## 11. Future Improvements

- **Gradient Boosting (XGBoost / LightGBM)** — Often outperforms Random Forest on tabular data.
- **Advanced imputation (MICE / k-NN)** — Models missing values using other features instead of simple median.
- **Deeper feature engineering** — Cabin position, ticket prefix, family surname survival correlations.
- **Bayesian / randomised hyperparameter search** — More efficient than grid search over large spaces.
- **Individual SHAP force plots** — Explain *per-passenger* predictions for richer interpretability.

---

## Presentation Flow (5–7 min)

1. Problem & data (38% survival, class imbalance)
2. Pipeline & features (no leakage)
3. Model comparison table + ROC / confusion matrix
4. Cross-validation results (robustness)
5. SHAP story (gender dominates; why RF for explanation, LR for deployment)
6. Live demo: `streamlit run app.py`
7. Limitations & future improvements
