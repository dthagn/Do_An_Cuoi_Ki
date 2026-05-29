"""Syntax check for new cells."""
import ast, json, sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

NB_PATH = Path("modeling.ipynb")
nb = json.load(NB_PATH.open("r", encoding="utf-8"))
cells = nb["cells"]

NEW_CELL_INDICES = [24, 26]

print("=== Syntax check for new code cells ===\n")
all_ok = True
for idx in NEW_CELL_INDICES:
    cell = cells[idx]
    src = "".join(cell["source"])
    print(f"--- Cell {idx} ---")
    print(src)
    try:
        ast.parse(src)
        print("OK: Syntax valid\n")
    except SyntaxError as e:
        print(f"ERROR: {e}\n")
        all_ok = False

print("All OK!" if all_ok else "Errors found!")
