# -*- coding: utf-8 -*-
"""
update_notebook_explanations.py
================================
Inserts/replaces markdown cells in modeling.ipynb to add
educational explanations for each visualization chart.

Run from the project root:
    py scripts/update_notebook_explanations.py
"""

import json
import sys
from pathlib import Path

NOTEBOOK_PATH = Path("modeling.ipynb")


# ─────────────────────────────────────────────────────────────────────────────
# New / replacement markdown cells to inject
# ─────────────────────────────────────────────────────────────────────────────

SECTION_HEADER_CELL = {
    "cell_type": "markdown",
    "metadata": {},
    "id": "1f774b07",
    "source": [
        "## 11. Model Evaluation — Charts & Explanations\n",
        "\n",
        "This section presents all evaluation visualizations for the Titanic survival prediction project.\n",
        "Four types of charts are generated:\n",
        "\n",
        "| Chart | Purpose | Model Used |\n",
        "|-------|---------|------------|\n",
        "| **Confusion Matrix** | Classify prediction errors (TP / TN / FP / FN) | Logistic Regression (best F1) |\n",
        "| **ROC Curve** | Compare discrimination ability across all thresholds | All 3 models |\n",
        "| **Feature Importance** | Rank features by predictive power (tree split gain) | Random Forest |\n",
        "| **SHAP Summary** | Explain how each feature impacts individual predictions | Random Forest |\n",
        "\n",
        "All charts are saved to `images/` at 180 dpi (presentation-ready).",
    ],
}

SHAP_INTRO_CELL = {
    "cell_type": "markdown",
    "metadata": {},
    "id": "778475bf",
    "source": [
        "### What is SHAP?\n",
        "\n",
        "**SHAP** (SHapley Additive exPlanations) is a game-theoretic method for explaining the output "
        "of any machine-learning model. It assigns each feature an *importance score* for a specific "
        "prediction, measured as the average contribution of that feature across all possible orderings.\n",
        "\n",
        "The **SHAP beeswarm plot** below shows:\n",
        "- Each **row** = one feature (sorted by mean absolute SHAP value — most important on top).\n",
        "- Each **dot** = one test-set passenger.\n",
        "- **Colour** = feature value (red = high, blue = low).\n",
        "- **Horizontal position** = impact on survival probability "
        "(right of 0 = increases survival, left of 0 = decreases survival).\n",
        "\n",
        "We use the **Random Forest** model for SHAP because tree-based models have an efficient exact SHAP "
        "algorithm (`TreeExplainer`), making computation fast and exact.\n",
        "\n",
        "> **Note:** Feature Importance (previous chart) measures how often a feature is used to split trees. "
        "SHAP measures the *direction* and *magnitude* of each feature's effect on each individual prediction. "
        "They complement each other.",
    ],
}

CONFUSION_MATRIX_EXPLANATION_CELL = {
    "cell_type": "markdown",
    "metadata": {},
    "id": "cm_explanation_01",
    "source": [
        "### Confusion Matrix — Reading Guide\n",
        "\n",
        "The confusion matrix below evaluates the **Logistic Regression** model (best F1 score on the hold-out set).\n",
        "\n",
        "| Quadrant | Meaning | Ideal? |\n",
        "|----------|---------|--------|\n",
        "| **TN** (top-left) | Correctly predicted *Not Survived* | Larger is better |\n",
        "| **FP** (top-right) | Predicted *Survived* but actually *Not Survived* | Smaller is better |\n",
        "| **FN** (bottom-left) | Predicted *Not Survived* but actually *Survived* | Smaller is better |\n",
        "| **TP** (bottom-right) | Correctly predicted *Survived* | Larger is better |\n",
        "\n",
        "**Key metrics derived from this matrix:**\n",
        "- **Precision** = TP / (TP + FP): Of all predicted survivors, how many truly survived?\n",
        "- **Recall** = TP / (TP + FN): Of all true survivors, how many did the model detect?\n",
        "- **F1-score** = harmonic mean of Precision and Recall — balances both.\n",
        "\n",
        "> **Observation:** The model correctly classifies ~84% of all test passengers "
        "(TN + TP cells). The dominant error type is False Negatives (FN), "
        "meaning some survivors were missed — a common challenge when the survival class is the minority.",
    ],
}

ROC_EXPLANATION_CELL = {
    "cell_type": "markdown",
    "metadata": {},
    "id": "roc_explanation_01",
    "source": [
        "### ROC Curve — Reading Guide\n",
        "\n",
        "The ROC (Receiver Operating Characteristic) curve plots **True Positive Rate** vs **False Positive Rate** "
        "across every possible classification threshold.\n",
        "\n",
        "| Concept | Explanation |\n",
        "|---------|-------------|\n",
        "| **TPR (Sensitivity)** | Fraction of actual survivors correctly identified |\n",
        "| **FPR (1 - Specificity)** | Fraction of non-survivors incorrectly flagged as survivors |\n",
        "| **Diagonal line** | Random classifier (AUC = 0.50) — our baseline |\n",
        "| **AUC** | Area Under the Curve — higher is better (max = 1.0) |\n",
        "\n",
        "**Model comparison:**\n",
        "- **Logistic Regression** achieves AUC ≈ 0.87 — best discriminating ability.\n",
        "- **Random Forest** achieves AUC ≈ 0.83 — strong, but slightly below LR.\n",
        "- **Decision Tree** achieves AUC ≈ 0.76 — lowest (overfitting to training splits).\n",
        "\n",
        "> **Why is Logistic Regression best here?** Despite being the simplest model, "
        "LR generalises well on small-to-medium tabular datasets with well-engineered features. "
        "The probability outputs are well-calibrated, which is precisely what ROC-AUC measures.",
    ],
}

FEATURE_IMPORTANCE_EXPLANATION_CELL = {
    "cell_type": "markdown",
    "metadata": {},
    "id": "fi_explanation_01",
    "source": [
        "### Feature Importance — Reading Guide\n",
        "\n",
        "Feature importance for **Random Forest** is measured by **Mean Decrease in Impurity (MDI)**: "
        "how much each feature reduces the Gini impurity when used for a tree split, averaged over all trees.\n",
        "\n",
        "**Top findings:**\n",
        "- **Age** (~19%) and **Fare** (~19%) are the two most predictive features — nearly equal weight.\n",
        "- **Title: Mr** (~15%) — the engineered title feature is highly informative: "
        "being male and adult strongly reduces survival probability.\n",
        "- **Sex** (~12%) — gender is the 4th most important feature. Combined with Title, "
        "gender-related signals account for ~27% of total importance.\n",
        "- **Group Size** and **Deck** also contribute, reflecting social context aboard the ship.\n",
        "\n",
        "> **Note:** MDI importance can be biased toward high-cardinality or continuous features (like Age, Fare). "
        "The SHAP plot (below) provides a complementary, less-biased view of each feature's true impact.",
    ],
}

SHAP_INTERPRETATION_CELL = {
    "cell_type": "markdown",
    "metadata": {},
    "id": "shap_interpretation_01",
    "source": [
        "### SHAP Summary — Interpretation\n",
        "\n",
        "Key patterns visible in the SHAP beeswarm:\n",
        "\n",
        "- **Title: Mr** (top feature): Red dots (male, adult title) cluster strongly to the **left** "
        "(SHAP < 0) — being a Mr dramatically *decreases* predicted survival probability. "
        "Blue dots (non-Mr title) shift right — other titles increase survival odds.\n",
        "\n",
        "- **Sex**: Blue dots (male, value=0) push left; red (female, value=1) push right. "
        "Confirms 'women and children first' — gender is a strong directional driver.\n",
        "\n",
        "- **Fare**: High-fare passengers (red, right side) have higher survival probability — "
        "a proxy for 1st class access to lifeboats.\n",
        "\n",
        "- **Age**: Mixed effect — very young (blue, children) tend to survive (right), "
        "older adults have lower probability (left).\n",
        "\n",
        "- **Pclass: 3**: High value = 3rd class (red dots, left) strongly reduces survival.\n",
        "\n",
        "- **Pclass: 1**: High value = 1st class (red dots, right) increases survival probability.\n",
        "\n",
        "> **Takeaway:** SHAP reveals that the Titanic survival model is fundamentally driven by "
        "**gender + social class + age** — consistent with historical accounts of evacuation priority.",
    ],
}

SURVIVAL_RATE_EXPLANATION_CELL = {
    "cell_type": "markdown",
    "metadata": {},
    "id": "sr_explanation_01",
    "source": [
        "### Survival Rate Distribution — Reading Guide\n",
        "\n",
        "This EDA chart (based on the full training dataset) shows survival statistics before modelling.\n",
        "\n",
        "**Panel A — Overall:**\n",
        "- Only **38.4%** of the 891 passengers survived.\n",
        "- The dataset is **imbalanced** (more non-survivors than survivors) — "
        "this is why accuracy alone is insufficient; F1-score is used for model selection.\n",
        "\n",
        "**Panel B — By Gender:**\n",
        "- **Female survival rate: 74.2%** vs **male: 18.9%**\n",
        "- This 4× gap is the single strongest signal in the data and directly reflects "
        "the 'women and children first' evacuation policy.\n",
        "\n",
        "**Panel C — By Passenger Class:**\n",
        "- **1st Class: 63.0%** > **2nd Class: 47.3%** > **3rd Class: 24.2%**\n",
        "- Survival probability almost perfectly tracks passenger class — "
        "1st class passengers had better access to lifeboats and were given evacuation priority.\n",
        "\n",
        "> **Modelling implication:** These EDA insights directly explain why Sex, Pclass, Fare, and Title "
        "appear as top features in both Feature Importance and SHAP plots. The ML model has learned "
        "the same patterns that historical analysis reveals.",
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Notebook patching logic
# ─────────────────────────────────────────────────────────────────────────────

def patch_notebook():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]

    # ── Step 1: Replace section header cell (id 1f774b07) ────────────────────
    replaced_header = False
    for i, cell in enumerate(cells):
        if cell.get("id") == "1f774b07":
            cells[i] = SECTION_HEADER_CELL
            replaced_header = True
            print("[OK] Replaced section 11 header cell.")
            break

    if not replaced_header:
        print("[!] Warning: section header cell (id=1f774b07) not found.")

    # ── Step 2: Replace SHAP intro markdown (id 778475bf) ────────────────────
    replaced_shap = False
    for i, cell in enumerate(cells):
        if cell.get("id") == "778475bf":
            cells[i] = SHAP_INTRO_CELL
            replaced_shap = True
            print("[OK] Replaced SHAP intro cell.")
            break

    if not replaced_shap:
        print("[!] Warning: SHAP intro cell (id=778475bf) not found.")

    # ── Step 3: Find the main visualization code cell and insert
    #    explanation cells AFTER it ──────────────────────────────────────────
    # The viz code cell is identified by containing "y_pred_best = best_pipe.predict"
    viz_cell_idx = None
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            src = "".join(cell.get("source", []))
            if "y_pred_best = best_pipe.predict" in src:
                viz_cell_idx = i
                break

    if viz_cell_idx is None:
        print("[!] Warning: Visualization code cell not found — explanation cells not inserted.")
        sys.exit(1)

    print(f"[OK] Found visualization code cell at index {viz_cell_idx}.")

    # Check if explanation cells already exist (idempotency)
    existing_ids = {cell.get("id") for cell in cells}
    already_inserted = "cm_explanation_01" in existing_ids

    if not already_inserted:
        # Insert explanation cells right after the viz code cell
        insert_pos = viz_cell_idx + 1
        new_cells = [
            CONFUSION_MATRIX_EXPLANATION_CELL,
            ROC_EXPLANATION_CELL,
            FEATURE_IMPORTANCE_EXPLANATION_CELL,
            SHAP_INTERPRETATION_CELL,
            SURVIVAL_RATE_EXPLANATION_CELL,
        ]
        for j, nc in enumerate(new_cells):
            cells.insert(insert_pos + j, nc)
        print(f"[OK] Inserted {len(new_cells)} explanation cells after index {viz_cell_idx}.")
    else:
        # Replace existing explanation cells in-place
        explanation_ids = {
            "cm_explanation_01":   CONFUSION_MATRIX_EXPLANATION_CELL,
            "roc_explanation_01":  ROC_EXPLANATION_CELL,
            "fi_explanation_01":   FEATURE_IMPORTANCE_EXPLANATION_CELL,
            "shap_interpretation_01": SHAP_INTERPRETATION_CELL,
            "sr_explanation_01":   SURVIVAL_RATE_EXPLANATION_CELL,
        }
        for i, cell in enumerate(cells):
            cid = cell.get("id")
            if cid in explanation_ids:
                cells[i] = explanation_ids[cid]
                print(f"[OK] Updated explanation cell: {cid}")

    nb["cells"] = cells

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print("\n[DONE] Notebook updated successfully.")


if __name__ == "__main__":
    patch_notebook()
