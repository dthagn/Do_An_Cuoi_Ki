# -*- coding: utf-8 -*-
"""
improve_visualizations.py
=========================
Regenerates all Titanic ML charts with improved aesthetics,
annotations, readability, and presentation-ready styling.

Run from the project root:
    python scripts/improve_visualizations.py
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import joblib

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_PATH   = Path("data/train.csv")
TEST_PATH    = Path("data/test.csv")
MODEL_PATH   = Path("model.pkl")
IMAGES_DIR   = Path("images")
IMAGES_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# ── Global style ──────────────────────────────────────────────────────────────
FONT_FAMILY  = "DejaVu Sans"
BG_COLOR     = "#F8F9FA"
SPINE_COLOR  = "#CCCCCC"

def apply_base_style(ax, title, xlabel="", ylabel=""):
    """Apply a consistent, academic look to any axes."""
    ax.set_facecolor(BG_COLOR)
    ax.figure.patch.set_facecolor("white")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COLOR)
    ax.tick_params(axis="both", labelsize=10, colors="#333333")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color=SPINE_COLOR)
    ax.set_axisbelow(True)


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1 — CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred, model_name="Best Model"):
    """
    Improved Confusion Matrix.
    - Two-colour scheme (soft red / soft green) for correct vs wrong cells.
    - Bold counts + percentage labels in each cell.
    - Quadrant labels: TN / FP / FN / TP.
    - Full human-readable axis labels.
    """
    cm   = confusion_matrix(y_true, y_pred)
    total = cm.sum()
    labels = ["Not Survived", "Survived"]

    # Custom colour map: wrong cells -> soft salmon, correct -> teal
    cmap_correct = "#2E8B8B"
    cmap_wrong   = "#C0392B"

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("white")

    # Draw cells manually for full colour control
    quadrant_colours = [
        [cmap_correct, cmap_wrong],   # row 0: TN, FP
        [cmap_wrong,   cmap_correct], # row 1: FN, TP
    ]
    quadrant_names = [
        ["TN", "FP"],
        ["FN", "TP"],
    ]

    for i in range(2):
        for j in range(2):
            colour = quadrant_colours[i][j]
            rect = mpatches.FancyBboxPatch(
                (j - 0.45, 1 - i - 0.45), 0.90, 0.90,
                boxstyle="round,pad=0.05",
                linewidth=1.5, edgecolor="white",
                facecolor=colour, zorder=2,
            )
            ax.add_patch(rect)

            count   = cm[i, j]
            pct     = count / total * 100
            q_label = quadrant_names[i][j]

            # Count (large)
            ax.text(j, 1 - i + 0.15, f"{count}",
                    ha="center", va="center",
                    fontsize=22, fontweight="bold", color="white", zorder=3)
            # Percentage
            ax.text(j, 1 - i - 0.10, f"({pct:.1f}%)",
                    ha="center", va="center",
                    fontsize=12, color="white", alpha=0.9, zorder=3)
            # Quadrant name (small, upper-left corner)
            ax.text(j - 0.38, 1 - i + 0.38, q_label,
                    ha="left", va="top",
                    fontsize=9, color="white", style="italic", zorder=3)

    # Axes formatting
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels[::-1], fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=13, labelpad=10)
    ax.set_ylabel("True Label", fontsize=13, labelpad=10)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=15, fontweight="bold", pad=16)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")

    # Legend patches
    correct_patch = mpatches.Patch(color=cmap_correct, label="Correct prediction")
    wrong_patch   = mpatches.Patch(color=cmap_wrong,   label="Incorrect prediction")
    ax.legend(handles=[correct_patch, wrong_patch],
              loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=2, fontsize=10, frameon=False)

    plt.tight_layout()
    out_path = IMAGES_DIR / "confusion_matrix.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_path.resolve()}")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2 — ROC CURVE
# ═══════════════════════════════════════════════════════════════════════════════
def plot_roc_curves(fitted_pipelines, X_test, y_test):
    """
    Improved ROC Curve.
    - Colourful, distinct lines per model.
    - AUC annotated inside the legend AND as a text badge.
    - Shaded area-under-curve for the best model.
    - Clean diagonal reference line.
    """
    palette = {
        "LogisticRegression": "#2E8B57",  # sea green
        "DecisionTree":       "#E67E22",  # carrot orange
        "RandomForest":       "#2980B9",  # steel blue
    }
    dash = {
        "LogisticRegression": "-",
        "DecisionTree":       "--",
        "RandomForest":       "-.",
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor("white")

    best_auc = -1
    best_name = ""
    for name, pipe in fitted_pipelines.items():
        y_prob = pipe.predict_proba(X_test)[:, 1]
        auc    = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        colour = palette.get(name, "#555555")
        ls     = dash.get(name, "-")
        ax.plot(fpr, tpr, color=colour, lw=2.2, linestyle=ls,
                label=f"{name}  (AUC = {auc:.3f})")

        if auc > best_auc:
            best_auc  = auc
            best_fpr  = fpr
            best_tpr  = tpr
            best_name = name

    # Shade best model
    ax.fill_between(best_fpr, best_tpr, alpha=0.08,
                    color=palette.get(best_name, "#2E8B57"))

    # Diagonal baseline
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Random Chance (AUC = 0.50)")

    # Styling
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.04])
    ax.set_xlabel("False Positive Rate  (1 − Specificity)", fontsize=12, labelpad=8)
    ax.set_ylabel("True Positive Rate  (Sensitivity / Recall)", fontsize=12, labelpad=8)
    ax.set_title("ROC Curves — Hold-Out Test Set", fontsize=15, fontweight="bold", pad=14)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color=SPINE_COLOR)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, color=SPINE_COLOR)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COLOR)
    ax.tick_params(labelsize=10)

    legend = ax.legend(loc="lower right", fontsize=10.5, framealpha=0.95,
                       edgecolor=SPINE_COLOR)

    # Annotation: Best model badge
    ax.annotate(f"Best: {best_name}\nAUC = {best_auc:.3f}",
                xy=(0.96, 0.42), xycoords="axes fraction",
                ha="right", va="top", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#DFFFD6",
                          edgecolor="#2E8B57", alpha=0.9))

    plt.tight_layout()
    out_path = IMAGES_DIR / "roc_curve.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_path.resolve()}")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3 — FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
def _clean_feat_name(raw: str) -> str:
    """Turn pipeline feature names like 'num__Age' → 'Age', 'cat__Sex_male' → 'Sex: male'."""
    raw = raw.replace("num__", "").replace("cat__", "")
    if "_" in raw:
        parts = raw.split("_", 1)
        return f"{parts[0]}: {parts[1]}"
    return raw


def plot_feature_importance(rf_pipeline):
    """
    Improved horizontal Feature Importance chart.
    - Colour gradient (low → high importance).
    - Clean human-readable feature names.
    - Percentage labels on bars.
    - Top-N only.
    """
    importances = rf_pipeline.named_steps["model"].feature_importances_
    feat_names  = rf_pipeline.named_steps["preprocess"].get_feature_names_out()

    ser = (
        pd.Series(importances, index=feat_names)
          .sort_values(ascending=False)
          .head(15)
          .sort_values(ascending=True)   # ascending for horizontal bar (bottom = most important)
    )

    # Clean labels
    clean_names = [_clean_feat_name(n) for n in ser.index]

    # Colour gradient
    norm_vals = (ser.values - ser.values.min()) / (ser.values.max() - ser.values.min() + 1e-9)
    cmap = plt.cm.get_cmap("Blues")
    colours = [cmap(0.35 + 0.6 * v) for v in norm_vals]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor("white")

    bars = ax.barh(clean_names, ser.values, color=colours,
                   edgecolor="white", linewidth=0.6, height=0.68)

    # Percentage labels
    total = ser.values.sum()
    for bar, val in zip(bars, ser.values):
        pct = val / total * 100
        ax.text(bar.get_width() + ser.values.max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}  ({pct:.1f}%)",
                va="center", fontsize=9, color="#333333")

    ax.set_xlim(0, ser.values.max() * 1.35)
    ax.set_title("Feature Importance — Random Forest Model",
                 fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Mean Decrease in Impurity (importance score)", fontsize=12, labelpad=8)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, color=SPINE_COLOR)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_edgecolor(SPINE_COLOR)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=9)

    # Highlight top feature
    top_name = clean_names[-1]
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.get_yticklabels()[-1].set_color("#1a6395")

    plt.tight_layout()
    out_path = IMAGES_DIR / "feature_importance.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_path.resolve()}")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 4 — SHAP SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
def _transform_for_model(pipeline, X_raw):
    X_fe  = pipeline.named_steps["features"].transform(X_raw)
    X_imp = pipeline.named_steps["imputer"].transform(X_fe)
    X_sex = pipeline.named_steps["sex_encode"].transform(X_imp)
    return pipeline.named_steps["preprocess"].transform(X_sex)


def plot_shap_summary(rf_pipeline, X_test):
    """
    Improved SHAP beeswarm plot.
    - Larger figure, bigger fonts.
    - Clean feature labels (strip pipeline prefixes).
    - Informative title and axis label.
    """
    try:
        import shap
    except ModuleNotFoundError:
        print("[!] shap not installed - skipping SHAP plot.  pip install shap")
        return

    X_matrix   = _transform_for_model(rf_pipeline, X_test)
    rf_model   = rf_pipeline.named_steps["model"]
    feat_names = rf_pipeline.named_steps["preprocess"].get_feature_names_out()
    clean_feat = [_clean_feat_name(n) for n in feat_names]

    explainer   = shap.TreeExplainer(rf_model)
    shap_values = explainer(X_matrix)

    # For binary classification, select class-1 explanation
    if len(shap_values.shape) == 3:
        sv_plot = shap_values[:, :, 1]
    else:
        sv_plot = shap_values

    # Assign clean feature names directly to the Explanation object
    import copy
    sv_named = copy.copy(sv_plot)
    sv_named.feature_names = clean_feat

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("white")
    plt.sca(ax)

    shap.plots.beeswarm(
        sv_named,
        max_display=15,
        show=False,
    )

    ax = plt.gca()
    ax.set_title("SHAP Summary — Global Feature Influence (Random Forest)",
                 fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("SHAP Value  (impact on model output: survival probability)",
                  fontsize=11, labelpad=8)
    ax.tick_params(axis="y", labelsize=10)
    plt.gcf().patch.set_facecolor("white")

    plt.tight_layout()
    out_path = IMAGES_DIR / "shap_summary.png"
    try:
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        print(f"[OK] Saved: {out_path.resolve()}")
    except OSError as err:
        print(f"[!] SHAP save error: {err}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5 — SURVIVAL RATE CHART (expanded: Sex + Pclass breakdown)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_survival_rate(df):
    """
    Improved Survival Rate visualisation.
    Panel A: Overall survival count with percentage labels.
    Panel B: Survival rate by Sex.
    Panel C: Survival rate by Passenger Class.
    """
    survived_colors = ["#C0392B", "#27AE60"]   # red = not survived, green = survived

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor("white")
    fig.suptitle("Survival Distribution — Titanic Dataset",
                 fontsize=16, fontweight="bold", y=1.01)

    # ── Panel A: Overall ──────────────────────────────────────────────────────
    ax = axes[0]
    counts = df["Survived"].value_counts().sort_index()
    bars   = ax.bar(["Not Survived", "Survived"], counts.values,
                    color=survived_colors, edgecolor="white", linewidth=0.8, width=0.55)
    total  = counts.sum()
    for bar, val in zip(bars, counts.values):
        pct = val / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.01,
                f"{val}\n({pct:.1f}%)",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    apply_base_style(ax, "Overall Survival Count", ylabel="Number of Passengers")
    ax.set_ylim(0, counts.max() * 1.22)
    ax.yaxis.grid(True, linestyle="--", alpha=0.45, color=SPINE_COLOR)

    # ── Panel B: By Sex ───────────────────────────────────────────────────────
    ax = axes[1]
    sex_survival = df.groupby("Sex")["Survived"].mean()
    bar_labels   = sex_survival.index.str.capitalize()
    sex_colors   = ["#2980B9", "#E91E8C"]  # blue=male, pink=female
    bars = ax.bar(bar_labels, sex_survival.values * 100,
                  color=sex_colors, edgecolor="white", linewidth=0.8, width=0.5)
    for bar, val in zip(bars, sex_survival.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{val*100:.1f}%",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    apply_base_style(ax, "Survival Rate by Gender", ylabel="Survival Rate (%)")
    ax.set_ylim(0, 105)

    # ── Panel C: By Pclass ────────────────────────────────────────────────────
    ax = axes[2]
    pclass_survival = df.groupby("Pclass")["Survived"].mean()
    class_labels    = [f"Class {c}" for c in pclass_survival.index]
    class_colors    = ["#8E44AD", "#2980B9", "#E67E22"]  # 1st purple, 2nd blue, 3rd orange
    bars = ax.bar(class_labels, pclass_survival.values * 100,
                  color=class_colors, edgecolor="white", linewidth=0.8, width=0.55)
    for bar, val in zip(bars, pclass_survival.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{val*100:.1f}%",
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    apply_base_style(ax, "Survival Rate by Passenger Class", ylabel="Survival Rate (%)")
    ax.set_ylim(0, 105)

    plt.tight_layout()
    out_path = IMAGES_DIR / "survival_rate.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out_path.resolve()}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("Loading data & model...")
    df       = pd.read_csv(TRAIN_PATH)
    df_test  = pd.read_csv(TEST_PATH)
    pipeline = joblib.load(MODEL_PATH)

    # Recreate train/test split exactly as in the notebook
    from sklearn.model_selection import train_test_split

    FEATURE_COLS = [c for c in df.columns if c != "Survived"]
    X = df[FEATURE_COLS]
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # ── Rebuild fitted pipelines (re-use the saved model as the best one) ────
    # We need all three models for the ROC curve.  Clone the best pipeline and
    # retrain the other two quickly so we can draw their ROC curves.
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    import copy

    def _clone_pipeline_with_model(base_pipe, model):
        """Replace the 'model' step in the pipeline."""
        steps = [(name, copy.deepcopy(step) if name != "model" else model)
                 for name, step in base_pipe.steps]
        return Pipeline(steps)

    fitted_pipelines = {}
    model_map = {
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=RANDOM_STATE),
        "DecisionTree":       DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    }
    for name, mdl in model_map.items():
        p = _clone_pipeline_with_model(pipeline, mdl)
        p.fit(X_train, y_train)
        fitted_pipelines[name] = p

    # Best model (LogisticRegression) predictions
    best_pipe  = fitted_pipelines["LogisticRegression"]
    y_pred     = best_pipe.predict(X_test)
    rf_pipeline = fitted_pipelines["RandomForest"]

    print("\nGenerating improved visualizations...\n")
    plot_confusion_matrix(y_test, y_pred, model_name="Logistic Regression")
    plot_roc_curves(fitted_pipelines, X_test, y_test)
    plot_feature_importance(rf_pipeline)
    plot_shap_summary(rf_pipeline, X_test)
    plot_survival_rate(df)

    print("\n[DONE] All visualizations saved to images/")


if __name__ == "__main__":
    main()
