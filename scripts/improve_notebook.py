"""
Script to insert new academic improvement cells into modeling.ipynb.
Run from the project root: py -3 scripts/improve_notebook.py
"""
import json
import uuid
from pathlib import Path

NB_PATH = Path("modeling.ipynb")

def new_id():
    return uuid.uuid4().hex[:8]

def md_cell(source_lines):
    """Create a markdown cell dict from a list of source strings."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines,
        "id": new_id(),
    }

def code_cell(source_lines):
    """Create a code cell dict from a list of source strings."""
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source_lines,
        "execution_count": None,
        "outputs": [],
        "id": new_id(),
    }

# ─────────────────────────────────────────────
# New cells definitions
# ─────────────────────────────────────────────

# TASK 7 – Classification explanation (insert after cell 13 "Train/Test Split")
md_classification_why = md_cell([
    "### Why Classification?\n",
    "\n",
    "Survival prediction is a **binary classification** problem: each passenger either survived (1) or did not survive (0). "
    "We choose classification — not regression — because the output is a discrete category, not a continuous value.\n",
    "\n",
    "**Why compare multiple models?**  \n",
    "No single algorithm is always best. By training Logistic Regression (linear), Decision Tree (rule-based), and Random Forest "
    "(ensemble), we can objectively select the model with the best generalisation performance on unseen data.\n",
    "\n",
    "**Why is the train/test split important?**  \n",
    "Evaluating a model on the same data it was trained on produces *optimistically biased* metrics — the model has already "
    "memorised those examples. A held-out test set gives an honest estimate of real-world performance.\n",
    "\n",
    "**What is data leakage?**  \n",
    "Data leakage occurs when information from the test set (or future data) leaks into training — for example, fitting "
    "a scaler on the whole dataset before splitting. We avoid this by wrapping all preprocessing steps inside a "
    "`sklearn.Pipeline` and fitting *after* the split.",
])

# TASK 7 – Random Forest explanation (insert after cell 15 "Model Training" heading)
md_rf_why = md_cell([
    "### Why does Random Forest often outperform a single Decision Tree?\n",
    "\n",
    "A single Decision Tree is prone to **overfitting**: it can memorise training patterns that don't generalise. "
    "Random Forest builds many trees on random sub-samples of data and features, then averages (or majority-votes) "
    "their predictions — a technique called **bagging**. This reduces variance and usually yields better results on "
    "unseen data. The trade-off is reduced interpretability, which is why we also use SHAP for explanation.",
])

# TASK 6 – SHAP explanation BEFORE shap code (insert before cell 22 "ROC Curve…")
md_shap_before = md_cell([
    "### What is SHAP?\n",
    "\n",
    "**SHAP** (SHapley Additive exPlanations) is a game-theoretic method for explaining the output of any machine-learning model. "
    "It assigns each feature an *importance score* for a specific prediction, measured as the average contribution of that feature "
    "across all possible orderings.\n",
    "\n",
    "The **SHAP summary plot** below shows:\n",
    "- Each row = one feature.\n",
    "- Each dot = one test sample.\n",
    "- **Colour** = feature value (red = high, blue = low).\n",
    "- **Horizontal position** = impact on model output (right = increases survival probability).\n",
    "\n",
    "We use the **Random Forest** model for SHAP because tree-based models have an efficient exact SHAP algorithm "
    "(`TreeExplainer`), making the computation fast and exact.",
])

# TASK 6 – SHAP explanation AFTER shap code
md_shap_after = md_cell([
    "### SHAP Interpretation\n",
    "\n",
    "**Key insight — Sex dominates:**  \n",
    "The `Sex_female` / `Sex_male` features consistently appear at the top of the SHAP summary. "
    "This reflects the historical *\"women and children first\"* evacuation policy on the Titanic: female passengers "
    "had a much higher survival rate (~74%) than male passengers (~19%). The ML model has learnt this statistical pattern "
    "from the training data.\n",
    "\n",
    "> **Important caveat:** The model is learning *correlations* in historical data, not causation. "
    "Gender-based predictions reflect the dataset's inherent bias, not a universal truth about survival.\n",
    "\n",
    "**Other strong features:**\n",
    "- `Pclass` — First-class passengers had priority access to lifeboats.\n",
    "- `Fare` — Correlated with class and location on the ship.\n",
    "- `Age` — Children were given evacuation priority.",
])

# TASK 1 – StratifiedKFold CV (insert after evaluation section, after cell 20)
md_skfold = md_cell([
    "## 10b. Stratified K-Fold Cross-Validation\n",
    "\n",
    "A single train/test split can be sensitive to the random seed. "
    "**StratifiedKFold** cross-validation repeatedly splits the *training* data into `k` folds, preserving the class ratio "
    "in each fold, and averages the results — giving a more robust estimate of model performance.",
])

code_skfold = code_cell([
    "from sklearn.model_selection import StratifiedKFold, cross_val_score\n",
    "\n",
    "CV_FOLDS = 5\n",
    "skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)\n",
    "\n",
    "print(f\"Stratified {CV_FOLDS}-Fold Cross-Validation on TRAINING data\\n\" + \"=\" * 50)\n",
    "cv_results = {}\n",
    "for name, pipe in fitted_pipelines.items():\n",
    "    acc_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='accuracy')\n",
    "    f1_scores  = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='f1')\n",
    "    cv_results[name] = {\n",
    "        'CV Accuracy (mean)': acc_scores.mean(),\n",
    "        'CV Accuracy (std)':  acc_scores.std(),\n",
    "        'CV F1 (mean)':       f1_scores.mean(),\n",
    "        'CV F1 (std)':        f1_scores.std(),\n",
    "    }\n",
    "    print(f\"{name}:\")\n",
    "    print(f\"  Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}\")\n",
    "    print(f\"  F1-score: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}\")\n",
    "    print()\n",
    "\n",
    "cv_df = pd.DataFrame(cv_results).T.round(4)\n",
    "display(cv_df)",
])

# TASK 2 – Model comparison table (insert after cell 20)
md_comparison = md_cell([
    "## 10c. Model Comparison Table\n",
    "\n",
    "Side-by-side comparison of all models on the **hold-out test set** (20% stratified split). "
    "The best score in each column is highlighted.",
])

code_comparison = code_cell([
    "comparison_df = pd.DataFrame(results_list).set_index('Model')\n",
    "comparison_df = comparison_df[['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']]\n",
    "comparison_df = comparison_df.round(4)\n",
    "\n",
    "print(\"Model Comparison — hold-out test set\\n\" + \"=\" * 50)\n",
    "display(comparison_df.style.highlight_max(axis=0, props='font-weight: bold; color: darkgreen;'))",
])

# TASK 4 – Limitations (insert after conclusion cell 25)
md_limitations = md_cell([
    "## 14. Limitations\n",
    "\n",
    "Understanding the limitations of a model is as important as its performance metrics.\n",
    "\n",
    "### 1. Small Dataset\n",
    "The Titanic training set contains only **891 passengers** — a very small dataset by modern ML standards. "
    "Small datasets increase the risk of overfitting and make it hard to estimate performance with high confidence.\n",
    "\n",
    "### 2. Missing Values\n",
    "Key features have significant missing data:\n",
    "- **Age**: ~20% missing — imputed using median per title group (a reasonable, but imperfect, strategy).\n",
    "- **Cabin**: ~77% missing — we extract only the deck letter; most records lack cabin info entirely.\n",
    "- **Embarked**: 2 missing values — filled with the mode.\n",
    "\n",
    "Imputation introduces assumptions that may not reflect reality.\n",
    "\n",
    "### 3. Gender Bias in the Dataset\n",
    "The dataset strongly reflects historical social norms (\"women and children first\"). "
    "Sex is the most predictive feature, which means the model will systematically predict different survival "
    "probabilities based on gender. This is a **historical bias** in the data, not a flaw in the model.\n",
    "\n",
    "### 4. Possible Overfitting\n",
    "Especially for the Decision Tree and Random Forest, there is a risk of overfitting the training data. "
    "GridSearchCV mitigates this via cross-validation tuning, but with only 891 samples, overfitting remains "
    "a concern. The held-out test set and cross-validation scores provide honest estimates.\n",
    "\n",
    "### 5. Historical Data — No Generalisation\n",
    "This model is trained on a specific historical event. It cannot generalise to other maritime disasters, "
    "or even to the Titanic passengers not represented in the dataset.",
])

# TASK 5 – Future Improvements (insert after Limitations)
md_future = md_cell([
    "## 15. Future Improvements\n",
    "\n",
    "The following improvements could further strengthen this project academically:\n",
    "\n",
    "### Gradient Boosting (XGBoost / LightGBM)\n",
    "Modern gradient boosting libraries such as **XGBoost** and **LightGBM** often outperform Random Forest "
    "on tabular data. They train sequentially, with each tree correcting the errors of the previous one, "
    "leading to lower bias and competitive variance.\n",
    "\n",
    "### Advanced Missing-Value Imputation\n",
    "Instead of median-per-group imputation, methods such as **k-NN imputation** or **iterative imputation** "
    "(MICE) model the missing values using the other features, potentially reducing imputation error.\n",
    "\n",
    "### Deeper Feature Engineering\n",
    "Features such as cabin position (fore vs. aft), ticket prefix, or interaction terms (e.g., `Sex × Pclass`) "
    "could provide additional signal. Name-based features (surname frequency) can also capture family survival correlations.\n",
    "\n",
    "### Better Hyperparameter Optimisation\n",
    "The current `GridSearchCV` explores a fixed grid. **Randomised search** or **Bayesian optimisation** "
    "(e.g., Optuna) explore the hyperparameter space more efficiently and can find better configurations "
    "with fewer evaluations.\n",
    "\n",
    "### Deeper Explainability\n",
    "Beyond global SHAP summaries, **individual SHAP force plots** or **LIME** explanations could show "
    "exactly *why* the model predicted survival or death for a specific passenger — useful for both "
    "debugging and stakeholder communication.",
])

# ─────────────────────────────────────────────
# Load notebook & insert cells
# ─────────────────────────────────────────────

nb = json.load(NB_PATH.open("r", encoding="utf-8"))
cells = nb["cells"]

# We insert in reverse order of position so earlier insertions don't
# shift the indices used for later insertions.

# ── After cell 25 (Conclusion): Limitations + Future Improvements ──
# Insert at index 26 (after conclusion at 25)
cells.insert(26, md_future)
cells.insert(26, md_limitations)

# ── After cell 22 (ROC/SHAP code): SHAP after explanation ──
# (after the original cell 22, now shifted by 2 → 24)
cells.insert(25, md_shap_after)

# ── Before cell 22 (ROC/SHAP code): SHAP before explanation ──
# (original cell 22, now shifted → 22)
cells.insert(22, md_shap_before)

# ── After cell 20 (evaluation code): CV + comparison table ──
# (original cell 20, now shifted → 20)
# Insert MD+code for comparison, then MD+code for CV
cells.insert(21, code_comparison)
cells.insert(21, md_comparison)
cells.insert(21, code_skfold)
cells.insert(21, md_skfold)

# ── After cell 15 (Model Training header): RF explanation ──
# (original cell 15, now shifted → 15)
cells.insert(16, md_rf_why)

# ── After cell 13 (Train/Test Split header): classification explanation ──
# (original cell 13, now shifted → 13)
cells.insert(14, md_classification_why)

nb["cells"] = cells
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Done! Total cells after: {len(cells)}")
