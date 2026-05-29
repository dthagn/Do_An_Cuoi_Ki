"""Fix the comparison table cell (cell 26) to match the actual variable name (results_df)."""
import json
from pathlib import Path

NB_PATH = Path("modeling.ipynb")
nb = json.load(NB_PATH.open("r", encoding="utf-8"))
cells = nb["cells"]

new_source = [
    "display_df = results_df.rename(columns={\n",
    '    "model": "Model",\n',
    '    "accuracy": "Accuracy",\n',
    '    "precision": "Precision",\n',
    '    "recall": "Recall",\n',
    '    "f1": "F1-Score",\n',
    '    "roc_auc": "ROC-AUC",\n',
    "}).set_index(\"Model\").round(4)\n",
    "\n",
    'print("Model Comparison — hold-out test set\\n" + "=" * 50)\n',
    'display(display_df.style.highlight_max(axis=0, props="font-weight: bold; color: darkgreen;"))',
]

cells[26]["source"] = new_source
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
print("Fixed cell 26")
print("".join(cells[26]["source"]))
