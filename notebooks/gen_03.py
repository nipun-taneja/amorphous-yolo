#!/usr/bin/env python3
"""Generator for notebooks/03_kvasir_eiou_vs_aeiou.ipynb
Run:  python notebooks/gen_03.py
"""
import json, pathlib

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": source}

cells = []

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — Title
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md(
"""# 03 · Kvasir-SEG: EIoU vs AEIoU Thorough Comparison

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nipun-taneja/amorphous-yolo/blob/main/notebooks/03_kvasir_eiou_vs_aeiou.ipynb)

## Abstract

This notebook is a focused companion to `02_full_loss_comparison.ipynb`.
Where notebook 02 benchmarks **all 7 IoU-based losses** on the DUO underwater dataset,
this notebook zooms in on a single research question:

> **Does AEIoU (Amorphous-EIoU) outperform EIoU on a second amorphous-object domain?**

We test this on **Kvasir-SEG** — 1 000 colonoscopy polyp images from the Simula
Research Lab. Polyps are biomedical objects with highly irregular, concave boundaries
and variable aspect ratios, making them an ideal second test case for the AEIoU
λ-rigidity hypothesis:

> *Amorphous labels are imprecise → down-weighting the shape penalty (λ < 1) should
> improve both accuracy and noise robustness.*

### What this notebook does
1. Downloads Kvasir-SEG and converts segmentation masks → YOLO bounding boxes
2. Builds three validation splits: **clean**, **low-noise** (σ=0.02), **high-noise** (σ=0.08)
3. Trains **33 models** (3 EIoU + 10 AEIoU λ values × 3 splits) using YOLO26n
4. Runs **7 quantitative analyses** and **3 validity checks**
5. Produces publication-ready figures saved to `experiments_kvasir/analysis/`

### Cross-dataset validation
If the optimal λ found here matches the optimal λ from notebook 02 (DUO), that is
strong evidence of a domain-agnostic setting for amorphous object detection.
"""
))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Environment Setup
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md(
"""## Section 1 · Environment Setup

**Requirements**
- Google Colab with T4 GPU (or better — A100 recommended for speed)
- ~2–3 hours total runtime for all 33 training runs on T4
- No API keys required — Kvasir-SEG downloads directly from Simula Research Lab

**Runtime estimate per run:** ~3–5 min on T4 (20 epochs, 640 px, nano model)
"""
))

cells.append(code(
"""# --- Install pinned dependencies
# ultralytics 8.4.9: confirmed working with yolo26n.pt and our monkey-patch
# wandb 0.24.1: optional experiment tracking (silently skipped if no API key)
!pip install --upgrade pip -q
!pip install -U "ultralytics==8.4.9" "wandb==0.24.1" -q
print("Dependencies installed.")
"""
))

cells.append(code(
"""# --- Idempotent git clone
# Safe to re-run: skips the clone if the repo is already present.
# The repo contains src/losses.py (EIoULoss, AEIoULoss) and the data/ yaml files.
import os, sys

REPO_PATH = "/content/amorphous-yolo"
if not os.path.exists(f"{REPO_PATH}/.git"):
    print("Cloning amorphous-yolo...")
    os.system(f"git clone https://github.com/nipun-taneja/amorphous-yolo.git {REPO_PATH}")
else:
    print("Repo already present — skipping clone.")

# Add project root to Python path so src.losses is importable
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

os.chdir(REPO_PATH)
print(f"Working directory: {os.getcwd()}")
"""
))

cells.append(code(
"""# --- All experiment constants (single source of truth for this notebook)
import math, time
from pathlib import Path
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR  = Path("/content/amorphous-yolo")
# Kvasir-SEG dataset root; the zip extracts to kvasir-seg/images/ and kvasir-seg/masks/
DATASET_ROOT = PROJECT_DIR / "datasets" / "kvasir-seg"
# Separate experiments dir from DUO (notebook 02) to avoid run-name collisions
EXPERIMENTS  = PROJECT_DIR / "experiments_kvasir"
ANALYSIS_DIR = EXPERIMENTS / "analysis"
MANIFEST_PATH = EXPERIMENTS / "manifest.json"
EXPERIMENTS.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# ── Google Drive persistence ───────────────────────────────────────────────────
DRIVE_ROOT        = Path("/content/drive/MyDrive/amorphous_yolo")
DRIVE_EXPERIMENTS = DRIVE_ROOT / "experiments_kvasir"
DRIVE_AVAILABLE   = False   # set to True by mount_drive() below

# ── Training hyper-parameters ─────────────────────────────────────────────────
EPOCHS   = 20     # Sufficient for convergence comparison at nano scale
IMGSZ    = 640    # Standard YOLO input resolution
DEVICE   = 0      # GPU 0; change to "cpu" for debugging
MODEL_PT = "yolo26n.pt"  # Nano model — fast iteration, still meaningful metrics

# ── Random seeds for statistical rigour ───────────────────────────────────────
# For quick development runs, keep SEEDS = [42].
# For publication, set SEEDS = [42, 123, 456] to get mean ± std across 3 seeds.
# Each seed produces a separate run: kvasir_yolo26n_{loss}_{split}_s{seed}_e{epochs}
SEEDS = [42]

# ── Standard baselines from src/losses.py ─────────────────────────────────────
# All 6 published IoU-family losses are compared against AEIoU.
# This ensures AEIoU is benchmarked against the FULL field, not just EIoU.
BASELINE_LOSS_NAMES = ["iou", "giou", "diou", "ciou", "eiou", "eciou"]

# ── AEIoU rigidity grid ───────────────────────────────────────────────────────
# λ=0.1 -> nearly pure center-alignment loss (shape penalty down-weighted 90%)
# λ=0.5 -> moderate trust in polyp extent labels
# λ=1.0 -> full size penalty active (normalisation still differs from EIoU — target
#           dims vs enclosing dims — so AEIoU(λ=1) != EIoU; used as cross-check)
AEIOU_RIGIDITIES = [round(x * 0.1, 1) for x in range(1, 11)]

def _fmt_r(r):
    # Format rigidity float for use in run names: 0.3 -> '0p3'
    return str(r).replace(".", "p")

# Master list of all loss keys (baselines + AEIoU grid)
ALL_LOSS_KEYS = BASELINE_LOSS_NAMES + [f"aeiou_r{_fmt_r(r)}" for r in AEIOU_RIGIDITIES]

# ── Dataset split configs ─────────────────────────────────────────────────────
SPLIT_CONFIGS = {
    "clean": PROJECT_DIR / "data" / "kvasir_seg.yaml",
    "low":   PROJECT_DIR / "data" / "kvasir_seg_low.yaml",
    "high":  PROJECT_DIR / "data" / "kvasir_seg_high.yaml",
}

# ── Visualisation palette ─────────────────────────────────────────────────────
PALETTE = {
    # Baselines — warm/neutral tones
    "iou":         "#888888",  # Grey — vanilla IoU
    "giou":        "#BC6C25",  # Brown — GIoU
    "diou":        "#606C38",  # Olive — DIoU
    "ciou":        "#DDA15E",  # Tan — CIoU (Ultralytics default)
    "eiou":        "#E63946",  # Red — EIoU
    "eciou":       "#9B2226",  # Dark red — ECIoU
    # AEIoU — cool tones (blue-green gradient by λ)
    "aeiou_r0p1":  "#023E8A",  # Navy — λ=0.1
    "aeiou_r0p2":  "#0077B6",  # Blue — λ=0.2
    "aeiou_r0p3":  "#00B4D8",  # Cyan — λ=0.3
    "aeiou_r0p4":  "#48CAE4",  # Light cyan — λ=0.4
    "aeiou_r0p5":  "#90E0EF",  # Pale blue — λ=0.5
    "aeiou_r0p6":  "#2A9D8F",  # Teal — λ=0.6
    "aeiou_r0p7":  "#52B788",  # Green — λ=0.7
    "aeiou_r0p8":  "#74C69D",  # Light green — λ=0.8
    "aeiou_r0p9":  "#95D5B2",  # Pale green — λ=0.9
    "aeiou_r1p0":  "#6A4C93",  # Purple — λ=1.0
}

# ── Human-readable labels for plots ───────────────────────────────────────────
LOSS_LABELS = {
    "iou": "IoU", "giou": "GIoU", "diou": "DIoU", "ciou": "CIoU",
    "eiou": "EIoU", "eciou": "ECIoU",
}
for r in AEIOU_RIGIDITIES:
    LOSS_LABELS[f"aeiou_r{_fmt_r(r)}"] = f"AEIoU {chr(955)}={r}"

n_baselines = len(BASELINE_LOSS_NAMES)
n_aeiou     = len(AEIOU_RIGIDITIES)
n_splits    = len(SPLIT_CONFIGS)
n_seeds     = len(SEEDS)
n_total     = (n_baselines + n_aeiou) * n_splits * n_seeds

print("Constants loaded.")
print(f"  Baselines:  {BASELINE_LOSS_NAMES}")
print(f"  AEIoU grid: {n_aeiou} values ({AEIOU_RIGIDITIES})")
print(f"  Splits:     {list(SPLIT_CONFIGS.keys())}")
print(f"  Seeds:      {SEEDS}")
print(f"  Total planned runs: {n_total}")
"""
))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Dataset Setup
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md(
"""## Section 2 · Kvasir-SEG Dataset

### Why Kvasir-SEG for AEIoU validation?

Kvasir-SEG (Jha et al., 2020) contains **1 000 colonoscopy images** of colorectal
polyps annotated with pixel-level segmentation masks. Key properties that make it
ideal for testing AEIoU:

| Property | Value |
|---|---|
| Images | 1 000 |
| Classes | 1 (polyp) |
| Annotation type | Segmentation mask → converted to bbox |
| Avg polyp area | ~20–60% of image |
| Shape variability | High — round, elongated, flat, pedunculated |
| Boundary regularity | Low — irregular, lobular, concave edges |

### Amorphousness argument
A gastroenterologist annotating a polyp draws a rough contour around a blob with
no fixed shape. The resulting bounding box extent is **annotation-dependent**, not
geometry-determined. Two annotators will produce different box sizes for the same
polyp. This is exactly the regime where λ < 1 in AEIoU should help:
setting λ=0.3 tells the loss "the box center is reliable, but the width/height is
only 30% trustworthy."

### Data preparation pipeline
```
kvasir-seg.zip  →  images/ + masks/
                          ↓  (Cell 8: cv2.findContours + boundingRect)
                     YOLO labels: 0 cx cy w h  (normalised)
                          ↓  (80/20 split)
                     train/  +  valid/
                          ↓  (Cell 9: Gaussian jitter)
                     valid_low/ (σ=0.02)  +  valid_high/ (σ=0.08)
```

**What to look for:** Cell 10 should show 800 train images and 200 val images in each
of the three splits. If counts differ, the split or download step failed.
"""
))

cells.append(code(
"""# --- WandB setup — proper login with project configuration
# WANDB_API_KEY must be set as a Colab secret (Secrets panel, left sidebar).
# If not set, WandB is disabled so training still runs without hanging.
import os, wandb

# Project name used for all runs in this notebook
WANDB_PROJECT = "amorphous-yolo-kvasir"

# Try to read API key from Colab secrets first (most secure), then fall back
# to an existing environment variable set before this cell ran.
try:
    from google.colab import userdata
    api_key = userdata.get("WANDB_API_KEY")
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
        print("WandB API key loaded from Colab secrets.")
except Exception:
    pass   # not in Colab or secret not set

if os.environ.get("WANDB_API_KEY"):
    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=False)
    print(f"WandB logged in. Project: {WANDB_PROJECT}")
else:
    os.environ["WANDB_MODE"] = "disabled"
    print("WANDB_API_KEY not found — WandB disabled.")
    print("To enable: add WANDB_API_KEY in Colab Secrets (key icon, left sidebar).")
"""
))

cells.append(md(
"""### Google Drive: Mount, Restore & Persist

Results are written to Drive **after every training run** so a Colab timeout
only loses the run currently in progress. On session restart, completed runs
are restored from Drive so the notebook resumes exactly where it left off.

**What to look for:** The restore step should print a list of run names copied
back from Drive. These runs will show `[SKIP]` in the training cells below,
meaning no computation is wasted repeating them.

**Setup:** Drive will be mounted automatically below. If it fails (e.g. running
locally), training still works — Drive sync is silently skipped.
"""
))

cells.append(code(
"""# --- Google Drive: mount + restore completed runs from previous sessions
import shutil

def mount_drive():
    # Mount Google Drive and set DRIVE_AVAILABLE = True if successful.
    global DRIVE_AVAILABLE
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        DRIVE_EXPERIMENTS.mkdir(parents=True, exist_ok=True)
        DRIVE_AVAILABLE = True
        print(f"Drive mounted. Backup dir: {DRIVE_EXPERIMENTS}")
    except Exception as e:
        print(f"Drive not available ({e}). Running without Drive persistence.")
        DRIVE_AVAILABLE = False
    return DRIVE_AVAILABLE


def restore_from_drive():
    # Copy completed runs from Drive back to local EXPERIMENTS dir.
    # Called once at session start so skip-on-existing logic works correctly
    # even after a Colab timeout/restart.
    if not DRIVE_AVAILABLE:
        return
    if not DRIVE_EXPERIMENTS.exists():
        print("No Drive backup found yet — starting fresh.")
        return
    restored = 0
    for drive_run in sorted(DRIVE_EXPERIMENTS.iterdir()):
        if not drive_run.is_dir():
            continue
        local_run = EXPERIMENTS / drive_run.name
        # Only restore runs that completed (have results.csv) and aren't already local
        if (drive_run / "results.csv").exists() and not (local_run / "results.csv").exists():
            shutil.copytree(str(drive_run), str(local_run), dirs_exist_ok=True)
            restored += 1
            print(f"  [RESTORE] {drive_run.name}")
    if restored == 0:
        print("Nothing to restore — local experiments are up to date.")
    else:
        print(f"Restored {restored} completed run(s) from Drive.")


mount_drive()
restore_from_drive()
"""
))

cells.append(code(
"""# --- Download Kvasir-SEG (idempotent)
# Source: Simula Research Lab official server — no API key required.
# File size: ~44 MB zip. Extracts to datasets/kvasir-seg/images/ and masks/.
import zipfile, urllib.request, shutil

kvasir_zip  = PROJECT_DIR / "datasets" / "kvasir-seg.zip"
images_dir  = DATASET_ROOT / "images"

if images_dir.exists() and len(list(images_dir.glob("*.jpg"))) > 900:
    print(f"Kvasir-SEG already present ({len(list(images_dir.glob('*.jpg')))} images) — skipping download.")
else:
    print("Downloading Kvasir-SEG (~44 MB)...")
    (PROJECT_DIR / "datasets").mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(
        "https://datasets.simula.no/downloads/kvasir-seg.zip",
        kvasir_zip,
    )
    print(f"  Downloaded → {kvasir_zip}")

    # Extract: creates kvasir-seg/ folder inside datasets/
    with zipfile.ZipFile(kvasir_zip, "r") as z:
        z.extractall(PROJECT_DIR / "datasets")
    print(f"  Extracted  → {DATASET_ROOT}")

    # Verify extracted structure
    n_imgs  = len(list((DATASET_ROOT / "images").glob("*.jpg")))
    n_masks = len(list((DATASET_ROOT / "masks").glob("*.jpg")))
    print(f"  Images: {n_imgs}  |  Masks: {n_masks}")
    assert n_imgs == n_masks == 1000, f"Expected 1000 pairs, got {n_imgs}/{n_masks}"
    print("Download complete.")
"""
))

cells.append(code(
"""# --- Convert segmentation masks -> YOLO bbox labels + 80/20 train/val split
# Kvasir-SEG provides PNG segmentation masks (white=polyp, black=background).
# We find the bounding box of all non-zero pixels using cv2.findContours,
# then write a YOLO label file: "0 cx cy w h" (all normalised to [0,1]).
#
# Each image has at most one polyp region (single-class dataset), so each
# label file contains exactly one line.
import cv2
import numpy as np
import random
import shutil

TRAIN_DIR  = DATASET_ROOT / "train"
VALID_DIR  = DATASET_ROOT / "valid"

def _mask_to_yolo_bbox(mask_path):
    # Load a Kvasir-SEG mask and return (cx, cy, w, h) normalised to [0,1].
    # Returns None if the mask is blank (no polyp found).
    # Read mask as grayscale; Kvasir masks are 3-channel JPEGs with white polyp
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    # Threshold to binary: pixel > 127 → foreground
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours of the polyp region(s)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None  # Blank mask — skip this image

    # Union bounding rect across all contours (handles multi-fragment masks)
    all_pts = np.vstack(contours)  # shape: [N_pts, 1, 2]
    x, y, w, h = cv2.boundingRect(all_pts)

    # Normalise by image dimensions (H, W)
    H, W = mask.shape
    cx = (x + w / 2) / W
    cy = (y + h / 2) / H
    bw = w / W
    bh = h / H

    # Clamp to [0, 1] for safety (rounding at image edges)
    cx, cy, bw, bh = [float(np.clip(v, 0.0, 1.0)) for v in [cx, cy, bw, bh]]
    return cx, cy, bw, bh


if (TRAIN_DIR / "images").exists() and len(list((TRAIN_DIR/"images").glob("*.jpg"))) >= 800:
    print("Train/val split already exists — skipping conversion.")
else:
    print("Converting masks → YOLO labels and creating train/val split...")
    for split_dir in [TRAIN_DIR, VALID_DIR]:
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)

    # Collect all image stems and shuffle for reproducible split
    all_imgs = sorted((DATASET_ROOT / "images").glob("*.jpg"))
    random.seed(42)
    random.shuffle(all_imgs)

    # 80% train, 20% val
    split_idx = int(0.8 * len(all_imgs))
    splits = {"train": all_imgs[:split_idx], "valid": all_imgs[split_idx:]}

    skipped = 0
    for split_name, img_list in splits.items():
        out_img_dir = DATASET_ROOT / split_name / "images"
        out_lbl_dir = DATASET_ROOT / split_name / "labels"
        for img_path in img_list:
            stem = img_path.stem
            # Kvasir masks share the same filename as images (jpg)
            mask_path = DATASET_ROOT / "masks" / f"{stem}.jpg"
            bbox = _mask_to_yolo_bbox(mask_path)
            if bbox is None:
                skipped += 1
                continue
            cx, cy, bw, bh = bbox
            # Copy image (hard copy so paths are self-contained)
            shutil.copy2(img_path, out_img_dir / img_path.name)
            # Write YOLO label: class_id cx cy w h
            (out_lbl_dir / f"{stem}.txt").write_text(
                f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
            )

    print(f"  Conversion complete. Skipped {skipped} blank masks.")
    print(f"  Train: {len(list((TRAIN_DIR/'images').glob('*.jpg')))} images")
    print(f"  Valid: {len(list((VALID_DIR/'images').glob('*.jpg')))} images")
"""
))

cells.append(code(
"""# --- Build noise-perturbed validation splits (idempotent)
# Applies Gaussian jitter to YOLO bbox coordinates in valid/labels/ to simulate
# annotation noise. Unlike DUO (polygon format), Kvasir labels are standard
# axis-aligned YOLO boxes so jitter is applied directly to (cx, cy, w, h).
#
# sigma_low  = 0.02: ≈2% of image dimension → mild annotation imprecision
# sigma_high = 0.08: ≈8% of image dimension → severe noise, stress-test
import numpy as np
import shutil
from pathlib import Path

SIGMA_LOW  = 0.02
SIGMA_HIGH = 0.08

def _jitter_label_file(src_lbl, dst_lbl, sigma, rng):
    \"\"\"Perturb all bbox coords in a YOLO label file by N(0, sigma) and clip to [0,1].\"\"\"
    lines = Path(src_lbl).read_text().strip().split("\\n")
    out = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        cls_id = parts[0]
        cx, cy, w, h = [float(v) for v in parts[1:5]]
        # Add independent Gaussian noise to each coordinate
        # Clamp: ensures boxes stay inside image boundaries
        cx = float(np.clip(cx + rng.normal(0, sigma), 0.0, 1.0))
        cy = float(np.clip(cy + rng.normal(0, sigma), 0.0, 1.0))
        w  = float(np.clip(w  + rng.normal(0, sigma), 0.01, 1.0))  # min 1% width
        h  = float(np.clip(h  + rng.normal(0, sigma), 0.01, 1.0))  # min 1% height
        out.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    Path(dst_lbl).write_text("\\n".join(out) + "\\n")


def build_kvasir_noise_splits(root: Path, seed: int = 42):
    \"\"\"Build valid_low/ and valid_high/ from valid/. Idempotent.\"\"\"
    src_img_dir = root / "valid" / "images"
    src_lbl_dir = root / "valid" / "labels"
    rng = np.random.default_rng(seed)

    for split_name, sigma in [("valid_low", SIGMA_LOW), ("valid_high", SIGMA_HIGH)]:
        dst_img_dir = root / split_name / "images"
        dst_lbl_dir = root / split_name / "labels"

        if dst_img_dir.exists() and len(list(dst_img_dir.glob("*.jpg"))) >= 200:
            print(f"  {split_name}: already exists — skipping.")
            continue

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(src_img_dir.glob("*.jpg")):
            stem = img_path.stem
            # Images are identical — only labels change
            # Use symlink where possible; fall back to copy on Windows
            dst_img = dst_img_dir / img_path.name
            if not dst_img.exists():
                try:
                    dst_img.symlink_to(img_path)
                except (OSError, NotImplementedError):
                    shutil.copy2(img_path, dst_img)

            src_lbl = src_lbl_dir / f"{stem}.txt"
            dst_lbl = dst_lbl_dir / f"{stem}.txt"
            if src_lbl.exists():
                _jitter_label_file(src_lbl, dst_lbl, sigma, rng)

        n = len(list(dst_img_dir.glob("*.jpg")))
        print(f"  {split_name}: {n} images (σ={sigma})")


build_kvasir_noise_splits(DATASET_ROOT)
print("Noise splits ready.")
"""
))

cells.append(code(
"""# --- Verify all three split configs exist and have correct image counts
import yaml

print(f"{'Split':<10} {'YAML':<40} {'Val images':>12}")
print("-" * 65)
for split_name, cfg_path in SPLIT_CONFIGS.items():
    assert cfg_path.exists(), f"Missing yaml: {cfg_path}"
    cfg = yaml.safe_load(cfg_path.read_text())
    val_dir = DATASET_ROOT / cfg["val"]
    n_imgs = len(list(val_dir.glob("*.jpg")))
    status = "✓" if n_imgs >= 200 else "✗ CHECK"
    print(f"{split_name:<10} {str(cfg_path.name):<40} {n_imgs:>10}  {status}")

# Training split (same for all configs)
n_train = len(list((DATASET_ROOT / "train" / "images").glob("*.jpg")))
print(f"\\nTrain images (shared): {n_train}")
assert n_train >= 800, f"Expected 800 train images, found {n_train}"
print("All splits verified.")
"""
))

cells.append(md(
"""### Kvasir-SEG Sample Images

The following cell displays 5 random training images with ground-truth bounding boxes
overlaid in green. Labels show normalised box dimensions.

**What to look for:**
- Polyps should vary in size from small (<10% of image area) to large (>50%)
- Some polyps are near-circular; others are elongated or irregular blobs
- No two polyps look the same — this confirms the amorphous-object assumption
- The GT box should tightly contain the polyp with minimal slack

If boxes look misaligned, the mask→bbox conversion in Cell 8 may have a coordinate error.
"""
))

cells.append(code(
"""# --- Visualise 5 random training images with GT bounding boxes
# This is a qualitative sanity check: do the labels look correct?
import cv2, random, numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def _draw_yolo_bbox(ax, label_path, img_w, img_h, color="lime", lw=2):
    \"\"\"Draw YOLO-format bbox(es) from a label file onto a matplotlib axis.\"\"\"
    if not Path(label_path).exists():
        return
    for line in Path(label_path).read_text().strip().split("\\n"):
        if not line.strip():
            continue
        _, cx, cy, bw, bh = [float(v) for v in line.split()]
        # Convert normalised (cx,cy,w,h) → pixel (x1,y1,w,h) for matplotlib
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        rect = patches.Rectangle(
            (x1, y1), bw * img_w, bh * img_h,
            linewidth=lw, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 4, f"polyp  w={bw:.2f} h={bh:.2f}",
                color=color, fontsize=7, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.5, pad=1))


train_imgs = sorted((DATASET_ROOT / "train" / "images").glob("*.jpg"))
sample = random.sample(train_imgs, 5)

fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig.suptitle("Kvasir-SEG — 5 random training images with GT bounding boxes",
             fontsize=13, fontweight="bold")

for ax, img_path in zip(axes, sample):
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    ax.imshow(img)
    lbl_path = DATASET_ROOT / "train" / "labels" / f"{img_path.stem}.txt"
    _draw_yolo_bbox(ax, lbl_path, W, H)
    ax.set_title(img_path.stem[:20], fontsize=8)
    ax.axis("off")

plt.tight_layout()
save_path = ANALYSIS_DIR / "00_sample_images.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")  # → experiments_kvasir/analysis/00_sample_images.png
plt.show()
print(f"Saved → {save_path}")
"""
))

cells.append(md(
"""**Figure 0 · Sample Kvasir-SEG Training Images**

*x-axis / y-axis:* pixel coordinates · *Green boxes:* ground-truth YOLO bounding boxes
*Labels:* show normalised width and height of each box

**Expected pattern:** Boxes of varying sizes and aspect ratios — small flat lesions
alongside large pedunculated polyps. Irregular polyp shapes that extend to one side of
the box are normal and confirm the amorphous assumption.
If all boxes appear the same size/shape, the mask conversion may have a bug.
"""
))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Theory
# ─────────────────────────────────────────────────────────────────────────────
cells.append(md(
r"""## Section 3 · Theory: IoU Loss Family & AEIoU

### The IoU Loss Family — Complete Comparison

All losses share $1 - \text{IoU}$ as the base overlap term. They differ in what
auxiliary penalties they add and how they normalise them.

$$L_{\text{IoU}}   = 1 - \text{IoU}$$
$$L_{\text{GIoU}}  = 1 - \text{IoU} + \frac{|C \setminus (A \cup B)|}{|C|}$$
$$L_{\text{DIoU}}  = 1 - \text{IoU} + \frac{\rho^2(b,b^{gt})}{c^2}$$
$$L_{\text{CIoU}}  = 1 - \text{IoU} + \frac{\rho^2}{c^2} + \alpha v$$
$$L_{\text{EIoU}}  = 1 - \text{IoU} + \frac{\rho^2}{c^2} + \frac{(p_w-t_w)^2}{c_w^2} + \frac{(p_h-t_h)^2}{c_h^2}$$
$$L_{\text{ECIoU}} = 1 - \text{IoU} + \frac{\rho^2}{c^2} + \frac{(p_w-t_w)^2}{\max(p_w,t_w)^2} + \frac{(p_h-t_h)^2}{\max(p_h,t_h)^2}$$
$$L_{\text{AEIoU}} = 1 - \text{IoU} + \frac{\rho^2}{c^2} + \lambda\!\left(\frac{(p_w-t_w)^2}{t_w^2} + \frac{(p_h-t_h)^2}{t_h^2}\right)$$

where $\rho^2$ = squared center distance, $c^2$ = enclosing box diagonal$^2$,
$v = \frac{4}{\pi^2}(\arctan\frac{t_w}{t_h} - \arctan\frac{p_w}{p_h})^2$ (CIoU aspect ratio).

### Size-Term Normaliser Comparison — The Core Question

| Loss | Size term | Normaliser | Scales with |
|---|---|---|---|
| IoU | None | — | Pure overlap |
| GIoU | Enclosing box fill | $\|C\|$ | Scene context |
| DIoU | Center only | — | No size penalty |
| CIoU | Aspect ratio $v$ | Implicit | w/h ratio, not absolute size |
| **EIoU** | $(p_w-t_w)^2 + (p_h-t_h)^2$ | **Enclosing dims** $c_w^2, c_h^2$ | Scene clutter |
| **ECIoU** | $(p_w-t_w)^2 + (p_h-t_h)^2$ | **max(pred, target)** $^2$ | Larger box |
| **AEIoU** | $(p_w-t_w)^2 + (p_h-t_h)^2$ | **Target dims** $t_w^2, t_h^2$ × $\lambda$ | Label only |

### Three normalisation philosophies

**EIoU (enclosing box):** A 10 px width error is penalised less if the enclosing
box is 200 px wide. Sensible for rigid objects in cluttered scenes.

**ECIoU (max of pred/target):** The penalty scales with the larger of the two boxes.
More stable than EIoU when the enclosing box is much larger than both boxes.

**AEIoU (target dims + λ):** A 10 px error on a 20 px polyp = 25% of target$^2$.
The same error on a 200 px polyp = 0.25%. This is *label-relative*, not context-relative.
The λ parameter additionally down-weights the entire size term to account for
annotation noise in amorphous-object labels.

### The λ-rigidity argument for polyps

A gastroenterologist annotating a polyp draws a rough contour. The resulting
bounding box extent depends on where they stopped drawing, not the polyp's true geometry.
Two clinicians produce different box widths/heights for the same polyp.

Setting λ < 1 says: *"I trust the center coordinates more than the extent."*
- λ = 0.1: near-pure center-alignment loss — ignore width/height error
- λ = 0.3: down-weight size penalty 70% — moderate scepticism about extent labels
- λ = 1.0: full size penalty — same strength as EIoU (but different normaliser)

**Prediction:** The optimal λ for Kvasir-SEG will be 0.2–0.4, matching the DUO
dataset result, suggesting a domain-agnostic optimal λ for amorphous objects.
"""
))

cells.append(code(
"""# --- Numerical comparison: all 7 losses on 3 synthetic scenarios
# Unit-test style: verify losses respond correctly to canonical configs.
import torch
from src.losses import IoULoss, GIoULoss, DIoULoss, CIoULoss, EIoULoss, ECIoULoss, AEIoULoss

baseline_fns = {
    "IoU": IoULoss(reduction="none"),   "GIoU": GIoULoss(reduction="none"),
    "DIoU": DIoULoss(reduction="none"), "CIoU": CIoULoss(reduction="none"),
    "EIoU": EIoULoss(reduction="none"), "ECIoU": ECIoULoss(reduction="none"),
}
aeiou_fns = {f"AEIoU lam={r}": AEIoULoss(rigidity=r, reduction="none")
             for r in [0.1, 0.3, 0.5, 1.0]}
all_fns = {**baseline_fns, **aeiou_fns}

scenarios = {
    "Perfect match":   (torch.tensor([[10.,10.,50.,50.]]), torch.tensor([[10.,10.,50.,50.]])),
    "Partial overlap": (torch.tensor([[10.,10.,40.,40.]]), torch.tensor([[20.,20.,50.,50.]])),
    "No overlap":      (torch.tensor([[10.,10.,30.,30.]]), torch.tensor([[40.,40.,60.,60.]])),
}

import pandas as pd
rows = []
for scenario_name, (pred, target) in scenarios.items():
    row = {"Scenario": scenario_name}
    for fn_name, fn in all_fns.items():
        row[fn_name] = fn(pred, target).item()
    rows.append(row)

df_theory = pd.DataFrame(rows).set_index("Scenario")
print(df_theory.round(4).to_string())

print("\\nExpected: loss decreases monotonically from 'No overlap' -> 'Perfect match'.")
print("AEIoU(lam=0.1) should be lowest on 'Partial overlap' (shape penalty suppressed).")
print("ECIoU should sit between EIoU and AEIoU(lam=1.0) on 'Partial overlap'.")
"""
))

cells.append(md(
"""**Table · Numerical Loss Comparison (All 7 Losses)**

*Rows:* three canonical prediction/target configurations
*Columns:* all 6 baselines + AEIoU at four representative λ values

**Expected pattern:**
- All losses = 0.0 on "Perfect match", highest on "No overlap"
- IoU and GIoU have no explicit size penalty — they only see overlap
- DIoU adds center distance but no size term — same as IoU when centers coincide
- CIoU's aspect-ratio term $v$ is non-zero when pred/target have different shapes
- EIoU, ECIoU, AEIoU(λ=1.0) differ only in normaliser — values should be close but not equal
- AEIoU(λ=0.1) gives the lowest loss on "Partial overlap" — size penalty nearly zero
"""
))

cells.append(code(
"""# --- Loss sensitivity sweep: slide from no-overlap to perfect overlap
# Constructs a 1D sweep where the predicted box moves from no overlap (left)
# to perfect overlap (right) with the target. Plots loss value vs IoU.
# This reveals the gradient landscape each loss provides to the optimiser.
import matplotlib.pyplot as plt
import torch
import numpy as np
from src.losses import IoULoss, GIoULoss, DIoULoss, CIoULoss, EIoULoss, ECIoULoss, AEIoULoss

target = torch.tensor([[40., 40., 80., 80.]])  # Fixed target: 40x40 px box

# Sweep: predicted box x-offset from -40 (no overlap) to 0 (perfect overlap)
offsets = np.linspace(-40, 0, 200)
sweep_fns = {
    "IoU":  IoULoss(reduction="none"),  "GIoU": GIoULoss(reduction="none"),
    "DIoU": DIoULoss(reduction="none"), "CIoU": CIoULoss(reduction="none"),
    "EIoU": EIoULoss(reduction="none"), "ECIoU": ECIoULoss(reduction="none"),
}
aeiou_sweep = {f"AEIoU lam={r}": AEIoULoss(rigidity=r, reduction="none")
               for r in [0.1, 0.3, 0.5, 1.0]}
all_sweep = {**sweep_fns, **aeiou_sweep}

ious = []
sweep_vals = {k: [] for k in all_sweep}

for dx in offsets:
    pred = torch.tensor([[dx + 40., 40., dx + 80., 80.]]).clamp(min=0)
    # Compute IoU for x-axis
    inter_x = max(0, min(pred[0,2].item(), target[0,2].item()) - max(pred[0,0].item(), target[0,0].item()))
    inter_y = max(0, min(pred[0,3].item(), target[0,3].item()) - max(pred[0,1].item(), target[0,1].item()))
    inter = inter_x * inter_y
    area_p = (pred[0,2]-pred[0,0]) * (pred[0,3]-pred[0,1])
    area_t = (target[0,2]-target[0,0]) * (target[0,3]-target[0,1])
    iou = inter / (float(area_p) + float(area_t) - inter + 1e-7)
    ious.append(float(iou))
    for k, fn in all_sweep.items():
        sweep_vals[k].append(float(fn(pred, target)))

# Plot baselines as solid lines, AEIoU as dashed
fig, ax = plt.subplots(figsize=(12, 6))
base_colors = {"IoU":"#888","GIoU":"#BC6C25","DIoU":"#606C38","CIoU":"#DDA15E","EIoU":"#E63946","ECIoU":"#9B2226"}
for k in sweep_fns:
    ax.plot(ious, sweep_vals[k], color=base_colors[k], lw=2, label=k)
aeiou_colors = {"AEIoU lam=0.1":"#023E8A","AEIoU lam=0.3":"#00B4D8","AEIoU lam=0.5":"#90E0EF","AEIoU lam=1.0":"#6A4C93"}
for k in aeiou_sweep:
    ax.plot(ious, sweep_vals[k], color=aeiou_colors[k], lw=1.8, linestyle="--", label=k)

ax.set_xlabel("IoU with target box", fontsize=12)
ax.set_ylabel("Loss value", fontsize=12)
ax.set_title("Loss sensitivity: no-overlap -> perfect overlap (all 7 loss families)",
             fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.invert_xaxis()  # left=no overlap (IoU=0), right=perfect (IoU=1)
ax.grid(alpha=0.3)
plt.tight_layout()
save_path = ANALYSIS_DIR / "01_loss_sensitivity.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")  # → experiments_kvasir/analysis/01_loss_sensitivity.png
plt.show()
print(f"Saved → {save_path}")
"""
))

cells.append(md(
"""**Figure 1 · Loss Sensitivity Sweep (All Loss Families)**

*x-axis:* IoU between predicted and target box (right = perfect overlap, left = no overlap)
*y-axis:* scalar loss value · *Solid lines:* baselines · *Dashed lines:* AEIoU variants

**Expected pattern:**
- All curves converge to 0 at IoU = 1.0 (rightmost point)
- IoU and GIoU have no explicit size term — flattest curves
- DIoU adds center distance but is identical to IoU when centers are aligned
- CIoU, EIoU, ECIoU add size penalties — highest curves at low IoU
- AEIoU(lam=0.1) has the smallest loss at IoU=0 (size penalty nearly zero)
- EIoU, ECIoU, and AEIoU(lam=1.0) should cluster together but not coincide
  (different normalisers = different gradient landscapes)
"""
))

print(f"Cells so far: {len(cells)}")  # progress check
