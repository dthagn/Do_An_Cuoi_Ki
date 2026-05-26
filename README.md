# Titanic Survival Prediction

Binary classification on the [Kaggle Titanic](https://www.kaggle.com/c/titanic) dataset: predict whether a passenger survived.

## Project structure

```
Do_An_Cuoi_Ki/
├── data/              train.csv, test.csv
├── images/            evaluation plots (ROC, confusion matrix, SHAP, …)
├── reports/
│   ├── report.docx    written report (Vietnamese)
│   └── slides.pptx    presentation deck
├── modeling.ipynb     full ML pipeline
├── app.py             Streamlit demo
├── model.pkl          trained pipeline (for app)
├── submission.csv     Kaggle submission
├── requirements.txt
└── README.md
```

Regenerate reports after re-running the notebook (new plots):

```bash
pip install python-docx python-pptx
python scripts/build_reports.py
```

## Dataset

- `data/train.csv` — 891 labeled passengers
- `data/test.csv` — 418 rows for Kaggle submission

## Features

Raw: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Name, Ticket, Cabin.

Engineered (inside the sklearn pipeline): Family_Size, Is_Alone, Title, Deck, Group_Size.

## Models

Three classifiers on a stratified 20% hold-out:

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|-----|---------|
| **Logistic Regression** | 0.8436 | **0.7879** | 0.8725 |
| Decision Tree | 0.8268 | 0.7634 | 0.8502 |
| Random Forest | 0.8101 | 0.7424 | 0.8449 |

- **Deploy / Kaggle:** best model by hold-out **F1** → refit on full train → `submission.csv`, `model.pkl` (currently Logistic Regression).
- **Explainability:** Random Forest + SHAP summary → `images/shap_summary.png` (shown in app and reports).

## How to run

```bash
pip install -r requirements.txt
# Run all cells in modeling.ipynb
streamlit run app.py
```

## Presentation flow (5–7 min)

1. Problem & data (38% survival, imbalance)
2. Pipeline & features (no leakage)
3. Model comparison table + ROC / confusion matrix
4. SHAP story (why RF for explanation, LR for deployment)
5. Live demo: `streamlit run app.py`
6. Limits & next steps

## Conclusion

End-to-end ML: EDA → cleaning → feature engineering → compare models → GridSearch → evaluation → Kaggle file → Streamlit demo → `reports/` for defense.
