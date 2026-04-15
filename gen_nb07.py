import json, sys
sys.stdout.reconfigure(encoding='utf-8')

def code_cell(source, cell_id):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [source]
    }

def md_cell(source, cell_id):
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source if isinstance(source, list) else [source]
    }

def src(text):
    lines = text.split('\n')
    if lines and lines[0] == '':
        lines = lines[1:]
    while lines and lines[-1] == '':
        lines.pop()
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + '\n')
        else:
            result.append(line)
    return result

cells = []

# Cell 0: Title
cells.append(md_cell(src(r"""
# 07 · Cross-Dataset Analysis: AEIoU Performance Summary

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nipun-taneja/amorphous-yolo/blob/main/notebooks/07_cross_dataset_analysis.ipynb)

**Purpose:** Load results from all experiment directories and produce paper-ready
cross-dataset comparison tables and figures.

**No training in this notebook.** All analysis is purely from saved `results.csv` files.

**Prerequisite notebooks:**
- `03_kvasir_eiou_vs_aeiou.ipynb` → `experiments_kvasir/`
- `04_isic2018.ipynb` → `experiments_isic/`
- `05_coco_amorphous_rigid.ipynb` → `experiments_coco_amorphous/`, `experiments_coco_rigid/`

**Figures produced:**
1. Unified mAP comparison table (paper Table 2)
2. AEIoU delta vs EIoU by dataset (amorphous vs rigid control)
3. Cross-dataset λ consistency plot
4. Per-dataset noise robustness comparison
"""), "cell-00-title"))

# Cell 1: Setup markdown
cells.append(md_cell(src(r"""
## Section 1 · Setup

This notebook loads results from Google Drive. Run the Drive mount cell first,
then load all datasets. Analysis cells can be re-run individually.

**Data sources (Drive paths):**
```
MyDrive/amorphous_yolo/experiments_kvasir/
MyDrive/amorphous_yolo/experiments_isic/
MyDrive/amorphous_yolo/experiments_coco_amorphous/
MyDrive/amorphous_yolo/experiments_coco_rigid/
```
"""), "cell-01-setup-md"))

# Cell 2: pip install
cells.append(code_cell(src(r"""
# --- Install minimal dependencies (no ultralytics needed — analysis only)
!pip install pandas matplotlib scipy seaborn -q
print("Dependencies installed.")
"""), "cell-02-install"))

# Cell 3: git clone
cells.append(code_cell(src(r"""
# --- Idempotent git clone (needed for PALETTE and LOSS_LABELS constants)
import os, sys

REPO_PATH = "/content/amorphous-yolo"
if not os.path.exists(f"{REPO_PATH}/.git"):
    os.system(f"git clone https://github.com/nipun-taneja/amorphous-yolo.git {REPO_PATH}")
else:
    print("Repo already present.")

if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
os.chdir(REPO_PATH)
print(f"CWD: {os.getcwd()}")
"""), "cell-03-clone"))

# Cell 4: constants
cells.append(code_cell(src(r"""
# --- Constants and Drive paths
from pathlib import Path

PROJECT_DIR = Path("/content/amorphous-yolo")

# ── Google Drive experiment directories ───────────────────────────────────────
DRIVE_ROOT = Path("/content/drive/MyDrive/amorphous_yolo")

DRIVE_DIRS = {
    "kvasir":         DRIVE_ROOT / "experiments_kvasir",
    "isic":           DRIVE_ROOT / "experiments_isic",
    "coco_amorphous": DRIVE_ROOT / "experiments_coco_amorphous",
    "coco_rigid":     DRIVE_ROOT / "experiments_coco_rigid",
}

# ── Local mirror (for analysis without Drive) ─────────────────────────────────
LOCAL_DIRS = {
    "kvasir":         PROJECT_DIR / "experiments_kvasir",
    "isic":           PROJECT_DIR / "experiments_isic",
    "coco_amorphous": PROJECT_DIR / "experiments_coco_amorphous",
    "coco_rigid":     PROJECT_DIR / "experiments_coco_rigid",
}

# ── Analysis output ───────────────────────────────────────────────────────────
ANALYSIS_DIR = PROJECT_DIR / "cross_dataset_analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset metadata ──────────────────────────────────────────────────────────
DATASET_META = {
    "kvasir":         {"label": "Kvasir-SEG",       "type": "amorphous", "n_classes": 1},
    "isic":           {"label": "ISIC 2018",         "type": "amorphous", "n_classes": 1},
    "coco_amorphous": {"label": "COCO Amorphous",    "type": "amorphous", "n_classes": 15},
    "coco_rigid":     {"label": "COCO Rigid",        "type": "rigid",     "n_classes": 10},
}

# ── Loss naming ───────────────────────────────────────────────────────────────
BASELINE_LOSS_NAMES = ["iou", "giou", "diou", "ciou", "eiou", "eciou", "siou", "wiou"]
AEIOU_RIGIDITIES    = [round(x * 0.1, 1) for x in range(1, 11)]

def _fmt_r(r):
    return str(r).replace(".", "p")

ALL_LOSS_KEYS = BASELINE_LOSS_NAMES + [f"aeiou_r{_fmt_r(r)}" for r in AEIOU_RIGIDITIES]

LOSS_LABELS = {
    "iou": "IoU", "giou": "GIoU", "diou": "DIoU", "ciou": "CIoU",
    "eiou": "EIoU", "eciou": "ECIoU", "siou": "SIoU", "wiou": "WIoU",
}
for r in AEIOU_RIGIDITIES:
    LOSS_LABELS[f"aeiou_r{_fmt_r(r)}"] = f"AEIoU \u03bb={r}"

PALETTE = {
    "iou":        "#888888", "giou":       "#BC6C25", "diou":       "#606C38",
    "ciou":       "#DDA15E", "eiou":       "#E63946", "eciou":      "#9B2226",
    "siou":       "#6A0572", "wiou":       "#FF6B6B",
    "aeiou_r0p1": "#023E8A", "aeiou_r0p2": "#0077B6", "aeiou_r0p3": "#00B4D8",
    "aeiou_r0p4": "#48CAE4", "aeiou_r0p5": "#90E0EF", "aeiou_r0p6": "#2A9D8F",
    "aeiou_r0p7": "#52B788", "aeiou_r0p8": "#74C69D", "aeiou_r0p9": "#95D5B2",
    "aeiou_r1p0": "#6A4C93",
}

DATASET_COLORS = {
    "kvasir":         "#2D6A4F",   # Dark green
    "isic":           "#1B4332",   # Darker green
    "coco_amorphous": "#0077B6",   # Blue
    "coco_rigid":     "#E63946",   # Red (control)
}

print("Constants loaded.")
print(f"  Datasets configured: {list(DATASET_META.keys())}")
print(f"  Analysis output: {ANALYSIS_DIR}")
"""), "cell-04-constants"))

# Cell 5: Drive mount
cells.append(code_cell(src(r"""
# --- Mount Google Drive and mirror results locally
import shutil

DRIVE_AVAILABLE = False


def mount_drive():
    global DRIVE_AVAILABLE
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        DRIVE_AVAILABLE = True
        print("Drive mounted.")
    except Exception as e:
        print(f"Drive not available ({e}). Using local results only.")
        DRIVE_AVAILABLE = False
    return DRIVE_AVAILABLE


def mirror_from_drive():
    # Copy results.csv and manifest.json from each Drive experiment dir
    # to the local mirror, so analysis works offline after the first fetch.
    if not DRIVE_AVAILABLE:
        print("Drive not mounted — using whatever local results exist.")
        return

    for ds_key, drive_dir in DRIVE_DIRS.items():
        local_dir = LOCAL_DIRS[ds_key]
        if not drive_dir.exists():
            print(f"  {ds_key}: no Drive directory found yet.")
            continue
        local_dir.mkdir(parents=True, exist_ok=True)
        n_mirrored = 0
        for drive_run in sorted(drive_dir.iterdir()):
            if not drive_run.is_dir():
                continue
            local_run = local_dir / drive_run.name
            if (drive_run / "results.csv").exists() and not (local_run / "results.csv").exists():
                shutil.copytree(str(drive_run), str(local_run), dirs_exist_ok=True)
                n_mirrored += 1
        # Also mirror manifest and combined CSV
        for fname in ["manifest.json", "all_results_combined.csv"]:
            src = drive_dir / fname
            if src.exists():
                shutil.copy2(str(src), str(local_dir / fname))
        print(f"  {ds_key}: {n_mirrored} runs mirrored.")


mount_drive()
mirror_from_drive()
"""), "cell-05-drive"))

# Cell 6: Load results from all datasets
cells.append(code_cell(src(r"""
# --- Load all results into a unified dictionary of DataFrames
import pandas as pd

MAP50_COL = "metrics/mAP50(B)"
MAP95_COL = "metrics/mAP50-95(B)"

# run_prefix map (prefix used in run names)
RUN_PREFIX = {
    "kvasir":         "kvasir",
    "isic":           "isic",
    "coco_amorphous": "coco_amorphous",
    "coco_rigid":     "coco_rigid",
}

SEEDS = [42]  # must match notebooks 03-05


def load_dataset_results(ds_key):
    # Load results.csv files for a dataset. Uses cache if available.
    local_dir = LOCAL_DIRS[ds_key]
    cache = local_dir / "all_results_combined.csv"

    if cache.exists():
        df = pd.read_csv(cache)
        print(f"  {ds_key}: {len(df)} rows from cache ({df['run_name'].nunique()} runs)")
        return df

    print(f"  {ds_key}: building from individual CSVs...")
    dfs = []
    prefix = RUN_PREFIX[ds_key]
    splits = ["clean", "low", "high"] if ds_key != "coco_rigid" else ["clean"]
    epochs = 20

    for loss_name in ALL_LOSS_KEYS:
        for split in splits:
            for seed in SEEDS:
                run_name = f"{prefix}_yolo26n_{loss_name}_{split}_s{seed}_e{epochs}"
                csv_path = local_dir / run_name / "results.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    df.columns = df.columns.str.strip()
                    df["run_name"] = run_name
                    df["loss"]     = loss_name
                    df["split"]    = split
                    df["seed"]     = seed
                    df["epoch"]    = df.index + 1
                    df["dataset"]  = ds_key
                    dfs.append(df)

    if not dfs:
        print(f"  {ds_key}: no results found.")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  {ds_key}: {len(combined)} rows ({len(dfs)} runs)")
    return combined


print("Loading results from all datasets...")
ALL_DFS = {}
for key in DATASET_META:
    ALL_DFS[key] = load_dataset_results(key)

available = [k for k, df in ALL_DFS.items() if not df.empty]
print(f"\nDatasets with results: {available}")
if len(available) < 2:
    print("[WARN] Need at least 2 datasets for cross-dataset analysis.")
"""), "cell-06-load-results"))

# Cell 7: Build unified summary
cells.append(code_cell(src(r"""
# --- Build unified summary: best AEIoU lambda and delta vs EIoU per dataset
import pandas as pd
import numpy as np

MAP95_COL = "metrics/mAP50-95(B)"
MAP50_COL = "metrics/mAP50(B)"

summary_rows = []

for ds_key, df in ALL_DFS.items():
    if df.empty:
        continue
    meta = DATASET_META[ds_key]
    df_final = df.groupby("run_name").last().reset_index()

    for split in ["clean", "low", "high"]:
        df_split = df_final[df_final["split"] == split]
        if df_split.empty:
            continue

        # EIoU baseline
        eiou_row = df_split[df_split["loss"] == "eiou"]
        eiou_95  = eiou_row[MAP95_COL].mean() if not eiou_row.empty and MAP95_COL in eiou_row.columns else float("nan")
        eiou_50  = eiou_row[MAP50_COL].mean() if not eiou_row.empty and MAP50_COL in eiou_row.columns else float("nan")

        # Best AEIoU lambda
        best_r, best_95, best_50 = None, -1, float("nan")
        for r in AEIOU_RIGIDITIES:
            k = f"aeiou_r{_fmt_r(r)}"
            sub = df_split[df_split["loss"] == k]
            if not sub.empty and MAP95_COL in sub.columns:
                m = sub[MAP95_COL].mean()
                if m > best_95:
                    best_95, best_r = m, r
                    best_50 = sub[MAP50_COL].mean() if MAP50_COL in sub.columns else float("nan")

        row = {
            "dataset":          ds_key,
            "dataset_label":    meta["label"],
            "dataset_type":     meta["type"],
            "n_classes":        meta["n_classes"],
            "split":            split,
            "eiou_map95":       eiou_95,
            "eiou_map50":       eiou_50,
            "best_aeiou_lambda": best_r,
            "best_aeiou_map95": best_95,
            "best_aeiou_map50": best_50,
            "delta_map95":      best_95 - eiou_95 if not np.isnan(eiou_95) and best_r is not None else float("nan"),
            "delta_map50":      best_50 - eiou_50 if not np.isnan(eiou_50) and best_r is not None else float("nan"),
        }
        summary_rows.append(row)

df_cross = pd.DataFrame(summary_rows)
df_cross.to_csv(ANALYSIS_DIR / "cross_dataset_summary.csv", index=False)

print("=== Cross-Dataset Summary (Clean Split) ===")
clean = df_cross[df_cross["split"] == "clean"].sort_values("dataset_type")
cols = ["dataset_label", "dataset_type", "eiou_map95", "best_aeiou_lambda", "best_aeiou_map95", "delta_map95"]
print(clean[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
"""), "cell-07-summary"))

# Cell 8: Section 2 markdown
cells.append(md_cell(src(r"""
## Section 2 · Unified Comparison Table (Paper Table 2)

Final mAP50-95 for EIoU and best AEIoU across all datasets and splits.
This is the paper's central results table.
"""), "cell-08-table-md"))

# Cell 9: Build paper table
cells.append(code_cell(src(r"""
# --- Paper Table 2: Unified comparison (EIoU vs best AEIoU, all datasets)
import pandas as pd
import numpy as np

if df_cross.empty:
    print("No cross-dataset results. Load results first.")
else:
    # Pivot: rows = dataset x split, cols = EIoU / best_AEIoU / delta
    rows_fmt = []
    for _, row in df_cross.sort_values(["dataset_type", "dataset", "split"]).iterrows():
        split_label = row["split"].capitalize()
        rows_fmt.append({
            "Dataset": row["dataset_label"],
            "Type": row["dataset_type"].capitalize(),
            "Split": split_label,
            "EIoU mAP50-95": f"{row['eiou_map95']:.4f}" if not np.isnan(row['eiou_map95']) else "-",
            "Best AEIoU mAP50-95": f"{row['best_aeiou_map95']:.4f}" if not np.isnan(row['best_aeiou_map95']) else "-",
            "Best \u03bb": str(row['best_aeiou_lambda']) if row['best_aeiou_lambda'] is not None else "-",
            "\u0394 mAP50-95": f"{row['delta_map95']:+.4f}" if not np.isnan(row['delta_map95']) else "-",
        })

    df_table = pd.DataFrame(rows_fmt)
    df_table.to_csv(ANALYSIS_DIR / "paper_table2.csv", index=False)

    print("=== Paper Table 2: EIoU vs Best AEIoU ===")
    print(df_table.to_string(index=False))
    print(f"\nSaved to: {ANALYSIS_DIR / 'paper_table2.csv'}")
"""), "cell-09-table"))

# Cell 10: Section 3 - Delta scatter
cells.append(md_cell(src(r"""
## Section 3 · AEIoU Delta vs EIoU: Amorphous vs Rigid

The key evidence figure: shows AEIoU provides larger gains on amorphous objects
than on rigid objects (control). A positive delta on amorphous + near-zero on rigid
confirms the amorphous-boundary hypothesis.
"""), "cell-10-delta-md"))

# Cell 11: Delta bar chart
cells.append(code_cell(src(r"""
# --- Fig 1: AEIoU delta (best lambda - EIoU) per dataset, clean split
import matplotlib.pyplot as plt
import numpy as np

if df_cross.empty:
    print("No data.")
else:
    clean = df_cross[df_cross["split"] == "clean"].copy()
    clean = clean.sort_values("delta_map95", ascending=True)

    colors = [DATASET_COLORS.get(row["dataset"], "#999") for _, row in clean.iterrows()]
    labels = [f"{row['dataset_label']}\n({row['dataset_type']})" for _, row in clean.iterrows()]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, clean["delta_map95"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("mAP50-95 \u0394 (best AEIoU \u2212 EIoU) on clean split")
    ax.set_title("Cross-Dataset AEIoU Benefit\nAmorphous datasets should show larger \u0394 than rigid control")

    # Add value labels
    for bar, val in zip(bars, clean["delta_map95"]):
        if not np.isnan(val):
            ax.text(val + 0.001 if val >= 0 else val - 0.001,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:+.4f}",
                    va="center", ha="left" if val >= 0 else "right", fontsize=9)

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(ANALYSIS_DIR / "fig1_delta_by_dataset.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("Fig 1 saved.")
"""), "cell-11-fig1"))

# Cell 12: Lambda consistency
cells.append(md_cell(src(r"""
## Section 4 · Cross-Dataset λ Consistency

If the optimal λ is consistently ≈0.3–0.5 across all amorphous datasets,
this supports using a fixed λ without per-dataset tuning.
"""), "cell-12-lambda-md"))

# Cell 13: Lambda consistency plot
cells.append(code_cell(src(r"""
# --- Fig 2: Optimal lambda across datasets (clean split)
import matplotlib.pyplot as plt
import numpy as np

if df_cross.empty:
    print("No data.")
else:
    clean = df_cross[df_cross["split"] == "clean"]
    amo   = clean[clean["dataset_type"] == "amorphous"]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = range(len(amo))
    lambdas = [row["best_aeiou_lambda"] for _, row in amo.iterrows()]
    labels  = [row["dataset_label"] for _, row in amo.iterrows()]
    colors  = [DATASET_COLORS.get(row["dataset"], "#999") for _, row in amo.iterrows()]

    bars = ax.bar(x, lambdas, color=colors, alpha=0.85, width=0.5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="\u03bb=0.5 reference")
    ax.axhspan(0.3, 0.5, alpha=0.1, color="blue", label="\u03bb=0.3\u20130.5 target range")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Optimal \u03bb (best mAP50-95 on clean split)")
    ax.set_ylim(0, 1.1)
    ax.set_title("Cross-Dataset Optimal \u03bb Consistency\n(Amorphous datasets only)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(ANALYSIS_DIR / "fig2_lambda_consistency.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("Fig 2 saved.")

    # Print summary
    lambdas_valid = [l for l in lambdas if l is not None]
    if lambdas_valid:
        print(f"\nOptimal \u03bb across amorphous datasets: {lambdas}")
        print(f"Mean: {np.mean(lambdas_valid):.2f}  Std: {np.std(lambdas_valid):.2f}")
        print(f"All in [0.3, 0.5]? {all(0.3 <= l <= 0.5 for l in lambdas_valid)}")
"""), "cell-13-fig2"))

# Cell 14: Lambda curves overlay
cells.append(code_cell(src(r"""
# --- Fig 3: Lambda-vs-mAP curves overlaid for all amorphous datasets
import matplotlib.pyplot as plt
import numpy as np

MAP95_COL = "metrics/mAP50-95(B)"

fig, ax = plt.subplots(figsize=(11, 7))

for ds_key, df in ALL_DFS.items():
    if df.empty or DATASET_META[ds_key]["type"] != "amorphous":
        continue
    meta = DATASET_META[ds_key]
    color = DATASET_COLORS[ds_key]
    df_final = df.groupby("run_name").last().reset_index()
    df_clean = df_final[df_final["split"] == "clean"]

    lambdas, maps = [], []
    for r in AEIOU_RIGIDITIES:
        k = f"aeiou_r{_fmt_r(r)}"
        sub = df_clean[df_clean["loss"] == k]
        if not sub.empty and MAP95_COL in sub.columns:
            lambdas.append(r)
            maps.append(sub[MAP95_COL].mean())

    if lambdas:
        ax.plot(lambdas, maps, "o-", color=color, linewidth=2.5,
                label=f"{meta['label']} (AEIoU)")

        # EIoU reference line
        eiou_sub = df_clean[df_clean["loss"] == "eiou"]
        if not eiou_sub.empty and MAP95_COL in eiou_sub.columns:
            eiou_val = eiou_sub[MAP95_COL].mean()
            ax.axhline(eiou_val, color=color, linestyle=":", alpha=0.5,
                       label=f"{meta['label']} EIoU = {eiou_val:.4f}")

# Add rigid control AEIoU for comparison
df_rig = ALL_DFS.get("coco_rigid", pd.DataFrame())
if not df_rig.empty:
    df_final_rig = df_rig.groupby("run_name").last().reset_index()
    df_clean_rig = df_final_rig[df_final_rig["split"] == "clean"]
    lambdas, maps = [], []
    for r in AEIOU_RIGIDITIES:
        k = f"aeiou_r{_fmt_r(r)}"
        sub = df_clean_rig[df_clean_rig["loss"] == k]
        if not sub.empty and MAP95_COL in sub.columns:
            lambdas.append(r)
            maps.append(sub[MAP95_COL].mean())
    if lambdas:
        ax.plot(lambdas, maps, "s--", color=DATASET_COLORS["coco_rigid"],
                linewidth=2.5, alpha=0.7, label="COCO Rigid — control (AEIoU)")

ax.axvspan(0.3, 0.5, alpha=0.08, color="blue")
ax.set_xlabel("\u03bb (AEIoU rigidity)", fontsize=12)
ax.set_ylabel("mAP50-95 (normalised per dataset)", fontsize=12)
ax.set_title("Cross-Dataset AEIoU \u03bb Curves — All Amorphous + Rigid Control\n"
             "Blue shaded region: target \u03bb=0.3\u20130.5", fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(str(ANALYSIS_DIR / "fig3_cross_dataset_lambda_curves.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Fig 3 saved.")
"""), "cell-14-fig3"))

# Cell 15: Noise robustness
cells.append(md_cell(src(r"""
## Section 5 · Cross-Dataset Noise Robustness

Compares how AEIoU and EIoU degrade under annotation noise (clean→high) across datasets.
"""), "cell-15-noise-md"))

# Cell 16: Noise robustness comparison
cells.append(code_cell(src(r"""
# --- Fig 4: Noise robustness — clean-high gap for EIoU vs best AEIoU across datasets
import matplotlib.pyplot as plt
import numpy as np

MAP95_COL = "metrics/mAP50-95(B)"

fig, ax = plt.subplots(figsize=(10, 6))

x_pos = 0
x_ticks = []
x_labels = []

for ds_key in ["kvasir", "isic", "coco_amorphous"]:
    df = ALL_DFS.get(ds_key, pd.DataFrame())
    if df.empty:
        continue
    meta = DATASET_META[ds_key]
    df_final = df.groupby("run_name").last().reset_index()

    # EIoU gap
    eiou_c = df_final[(df_final["loss"] == "eiou") & (df_final["split"] == "clean")]
    eiou_h = df_final[(df_final["loss"] == "eiou") & (df_final["split"] == "high")]
    eiou_gap = (eiou_c[MAP95_COL].mean() - eiou_h[MAP95_COL].mean()
                if not eiou_c.empty and not eiou_h.empty and MAP95_COL in eiou_c.columns
                else float("nan"))

    # Best AEIoU gap
    best_r = None
    best_map = -1
    for r in AEIOU_RIGIDITIES:
        k = f"aeiou_r{_fmt_r(r)}"
        sub = df_final[(df_final["loss"] == k) & (df_final["split"] == "clean")]
        if not sub.empty and MAP95_COL in sub.columns and sub[MAP95_COL].mean() > best_map:
            best_map = sub[MAP95_COL].mean()
            best_r = r

    aeiou_gap = float("nan")
    if best_r is not None:
        k = f"aeiou_r{_fmt_r(best_r)}"
        aeiou_c = df_final[(df_final["loss"] == k) & (df_final["split"] == "clean")]
        aeiou_h = df_final[(df_final["loss"] == k) & (df_final["split"] == "high")]
        if not aeiou_c.empty and not aeiou_h.empty and MAP95_COL in aeiou_c.columns:
            aeiou_gap = aeiou_c[MAP95_COL].mean() - aeiou_h[MAP95_COL].mean()

    bar_w = 0.35
    if not np.isnan(eiou_gap):
        ax.bar(x_pos, eiou_gap, bar_w, color=PALETTE["eiou"], alpha=0.85, label="EIoU" if x_pos == 0 else "")
    if not np.isnan(aeiou_gap):
        ax.bar(x_pos + bar_w, aeiou_gap, bar_w, color=DATASET_COLORS[ds_key], alpha=0.85,
               label=f"Best AEIoU" if x_pos == 0 else "")

    x_ticks.append(x_pos + bar_w / 2)
    x_labels.append(meta["label"])
    x_pos += 1.2

ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=10)
ax.set_ylabel("mAP50-95 drop (clean \u2212 high noise)")
ax.set_title("Cross-Dataset Noise Robustness: EIoU vs Best AEIoU\n"
             "Smaller gap = more robust to annotation noise")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(str(ANALYSIS_DIR / "fig4_cross_dataset_noise_robustness.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Fig 4 saved.")
"""), "cell-16-fig4"))

# Cell 17: Per-dataset mAP comparison
cells.append(code_cell(src(r"""
# --- Fig 5: All-loss mAP50-95 comparison (clean split) — all datasets stacked
import matplotlib.pyplot as plt
import numpy as np

MAP95_COL = "metrics/mAP50-95(B)"

available_datasets = [k for k, df in ALL_DFS.items() if not df.empty]
n_datasets = len(available_datasets)

if n_datasets == 0:
    print("No data available.")
else:
    fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 7), sharey=False)
    if n_datasets == 1:
        axes = [axes]

    for ax, ds_key in zip(axes, available_datasets):
        df = ALL_DFS[ds_key]
        meta = DATASET_META[ds_key]
        df_final = df.groupby("run_name").last().reset_index()
        df_clean = df_final[df_final["split"] == "clean"]

        vals = []
        for k in ALL_LOSS_KEYS:
            sub = df_clean[df_clean["loss"] == k]
            vals.append(sub[MAP95_COL].mean() if not sub.empty and MAP95_COL in sub.columns else 0.0)

        x = range(len(ALL_LOSS_KEYS))
        colors = [PALETTE.get(k, "#999") for k in ALL_LOSS_KEYS]
        ax.bar(x, vals, color=colors, alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels([LOSS_LABELS.get(k, k) for k in ALL_LOSS_KEYS],
                            rotation=55, ha="right", fontsize=7)
        ax.set_ylabel("mAP50-95")
        ax.set_title(f"{meta['label']}\n({meta['type']})", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("mAP50-95 by Loss — All Datasets (Clean Split)", fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(str(ANALYSIS_DIR / "fig5_all_datasets_map_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Fig 5 saved.")
"""), "cell-17-fig5"))

# Cell 18: Section 6 statistical summary
cells.append(md_cell(src(r"""
## Section 6 · Statistical Summary

Aggregate view: average AEIoU delta across all amorphous vs rigid datasets.
"""), "cell-18-stats-md"))

# Cell 19: Statistical summary
cells.append(code_cell(src(r"""
# --- Statistical summary: amorphous vs rigid average delta
import numpy as np
import pandas as pd

if df_cross.empty:
    print("No data.")
else:
    clean = df_cross[df_cross["split"] == "clean"]

    amo_deltas = clean[clean["dataset_type"] == "amorphous"]["delta_map95"].dropna().values
    rig_deltas = clean[clean["dataset_type"] == "rigid"]["delta_map95"].dropna().values

    print("=== Cross-Dataset AEIoU Delta Summary ===\n")
    print("Amorphous datasets (expected positive delta):")
    for _, row in clean[clean["dataset_type"] == "amorphous"].iterrows():
        d = row['delta_map95']
        lam = row['best_aeiou_lambda']
        print(f"  {row['dataset_label']:<20} best \u03bb={lam}  \u0394={d:+.4f}")
    if len(amo_deltas) > 0:
        print(f"  Average: {amo_deltas.mean():+.4f}  Std: {amo_deltas.std():.4f}")

    print("\nRigid datasets (expected ~zero delta):")
    for _, row in clean[clean["dataset_type"] == "rigid"].iterrows():
        d = row['delta_map95']
        lam = row['best_aeiou_lambda']
        print(f"  {row['dataset_label']:<20} best \u03bb={lam}  \u0394={d:+.4f}")
    if len(rig_deltas) > 0:
        print(f"  Average: {rig_deltas.mean():+.4f}  Std: {rig_deltas.std():.4f}")

    if len(amo_deltas) > 0 and len(rig_deltas) > 0:
        print(f"\nAmorphous avg delta ({amo_deltas.mean():+.4f}) vs Rigid ({rig_deltas.mean():+.4f})")
        print(f"Ratio: {abs(amo_deltas.mean()) / max(abs(rig_deltas.mean()), 1e-6):.1f}x larger on amorphous")

    # Optimal lambda consistency
    amo_lambdas = [row["best_aeiou_lambda"] for _, row in clean[clean["dataset_type"] == "amorphous"].iterrows()
                   if row["best_aeiou_lambda"] is not None]
    if amo_lambdas:
        print(f"\nOptimal \u03bb across amorphous datasets: {amo_lambdas}")
        print(f"Mean \u03bb = {np.mean(amo_lambdas):.2f} (target: 0.3\u20130.5)")
"""), "cell-19-stats"))

# Cell 20: Save and summary
cells.append(code_cell(src(r"""
# --- Save all analysis artifacts
import json

print("=== Cross-Dataset Analysis Artifacts ===\n")
print(f"Output dir: {ANALYSIS_DIR}")
print("\nFigures:")
for f in sorted(ANALYSIS_DIR.glob("*.png")):
    print(f"  {f.name:<55} {f.stat().st_size/1024:>6.1f} KB")
print("\nTables:")
for f in sorted(ANALYSIS_DIR.glob("*.csv")):
    print(f"  {f.name}")

print("\n\u2713 Cross-dataset analysis complete.")
print("The following figures are paper-ready:")
print("  fig1_delta_by_dataset.png        — Table 3 companion (AEIoU benefit by dataset)")
print("  fig2_lambda_consistency.png      — Optimal \u03bb stability across datasets")
print("  fig3_cross_dataset_lambda_curves.png — Lambda curves for all amorphous + rigid")
print("  fig4_cross_dataset_noise_robustness.png — Noise robustness EIoU vs AEIoU")
print("  fig5_all_datasets_map_comparison.png   — All-loss mAP comparison (supplement)")
print("  paper_table2.csv                 — Unified results table")
"""), "cell-20-save"))

# Cell 21: Final Drive sync
cells.append(code_cell(src(r"""
# --- Sync analysis to Drive
import shutil

if DRIVE_AVAILABLE:
    drive_analysis = DRIVE_ROOT / "cross_dataset_analysis"
    drive_analysis.mkdir(parents=True, exist_ok=True)
    shutil.copytree(str(ANALYSIS_DIR), str(drive_analysis), dirs_exist_ok=True)
    n_files = len(list(drive_analysis.rglob("*")))
    print(f"Analysis synced to Drive: {drive_analysis}")
    print(f"  {n_files} files backed up.")
else:
    print("Drive not mounted — analysis saved locally only.")
    print(f"  Local path: {ANALYSIS_DIR}")
"""), "cell-21-sync"))

# Build notebook
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": []},
    },
    "cells": cells
}

out_path = "C:/Users/PINCstudent/Downloads/SFSU/899/project/amorphous-yolo/notebooks/07_cross_dataset_analysis.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {out_path}")
print(f"Total cells: {len(cells)}")
