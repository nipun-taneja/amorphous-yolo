#!/usr/bin/env python3
"""
Combines gen_03.py (cells 0-15) and gen_03b.py (cells 16-49)
and writes the final notebook.

Run from repo root:
    python notebooks/gen_03_run.py
"""
import json, sys
from pathlib import Path

# ── Part 1: build cells 0-15 ─────────────────────────────────────────────────
# gen_03.py defines cells as a module-level list and prints a progress line.
# We exec it in an isolated namespace to capture the cells list.
part1_path = Path(__file__).parent / "gen_03.py"
ns1 = {}
exec(compile(part1_path.read_text(encoding='utf-8'), str(part1_path), "exec"), ns1)
cells = ns1["cells"]
print(f"Part 1: {len(cells)} cells loaded.")

# ── Part 2: append training cells ────────────────────────────────────────────
part2_path = Path(__file__).parent / "gen_03b.py"
ns2 = {}
exec(compile(part2_path.read_text(encoding='utf-8'), str(part2_path), "exec"), ns2)
add_cells_part2 = ns2["add_cells_part2"]
add_cells_part2(cells)
print(f"Part 2: total {len(cells)} cells after append.")

# ── Part 3: append analysis cells ────────────────────────────────────────────
part3_path = Path(__file__).parent / "gen_03c.py"
ns3 = {}
exec(compile(part3_path.read_text(encoding='utf-8'), str(part3_path), "exec"), ns3)
add_cells_part3 = ns3["add_cells_part3"]
add_cells_part3(cells)
print(f"Part 3: total {len(cells)} cells after append.")

# ── Assemble notebook JSON ────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "file_extension": ".py"
        },
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        }
    },
    "cells": cells,
}

out_path = Path(__file__).parent / "03_kvasir_eiou_vs_aeiou.ipynb"
out_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding='utf-8')
print(f"\nWritten: {out_path}  ({out_path.stat().st_size // 1024} KB, {len(cells)} cells)")
