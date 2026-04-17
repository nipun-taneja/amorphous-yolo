"""
Generate 05_cvc_colondb_kaggle.ipynb — a no-touch Kaggle notebook for
CVC-ColonDB that incorporates all fixes developed for the CVC-ClinicDB notebook.
Run this script locally: python generate_colondb_notebook.py
"""
import json, os

def _id(tag):
    return tag[:20]

def code(src, cid):
    return {"cell_type": "code", "execution_count": None,
            "id": _id(cid), "metadata": {}, "outputs": [],
            "source": src}

def md(src, cid):
    return {"cell_type": "markdown", "id": _id(cid),
            "metadata": {}, "source": src}

cells = []

# ── Cell 0: Title ──────────────────────────────────────────────────────────────
cells.append(md(
"""# 05 · CVC-ColonDB: EIoU vs AEIoU Thorough Comparison (Kaggle Edition)

Kaggle-adapted experiment for **CVC-ColonDB** polyp detection dataset (380 images).
Mirrors the methodology of `04_cvc_clinicdb_kaggle.ipynb` with all Kaggle-specific
path and compatibility fixes applied.

**Before running:**
1. Upload CVC-ColonDB as a Kaggle dataset (if not already done):
   - Go to kaggle.com → Datasets → New Dataset
   - Upload a zip containing `images/` and `masks/` folders (PNG files 1–380)
   - Name it e.g. `cvc-colondb`
   - Attach it here: *Add Data → Your Datasets → cvc-colondb → Add*
2. Enable GPU: *Settings → Accelerator → GPU T4 x2 (or P100)*
3. Run All Cells — fully automatic, no further input needed.

**Output:** `/kaggle/working/amorphous-yolo/experiments_colondb/`
Download via the *Output* tab after the run completes (~3–4 hours on T4).

| Property | Value |
|---|---|
| Total images | 380 |
| Train / Val  | ~304 / ~76 (80/20 random split, seed 42) |
| Classes | 1 (polyp) |
| Noise splits | clean, low (σ=0.02), high (σ=0.08) |
| Total planned runs | 18 losses × 3 splits = **54 runs** |
""", "colondb-title"))

# ── Cell 1: GPU check ──────────────────────────────────────────────────────────
cells.append(code(
"""# --- Kaggle: no Drive mount needed
print('Running on Kaggle. Output dir: /kaggle/working/')
import os
print('GPU:', os.popen('nvidia-smi --query-gpu=name --format=csv,noheader').read().strip())
""", "colondb-gpu-check"))

# ── Cell 2: Section 1 markdown ─────────────────────────────────────────────────
cells.append(md(
"""## Section 1 · Environment Setup

**Requirements**
- Kaggle Notebook with T4 GPU
- CVC-ColonDB dataset attached as input (upload your own or find on Kaggle)
- Internet enabled (for pip install + git clone)
""", "colondb-sec1-md"))

# ── Cell 3: Install deps ───────────────────────────────────────────────────────
cells.append(code(
"""# --- Install pinned dependencies
# ultralytics 8.4.9: confirmed working with yolo26n.pt and our monkey-patch
!pip install --upgrade pip -q
!pip install -U "ultralytics==8.4.9" "wandb==0.24.1" "pycocotools" -q
print("Dependencies installed.")
""", "colondb-install"))

# ── Cell 4: Git clone ──────────────────────────────────────────────────────────
cells.append(code(
"""# --- Idempotent git clone
import os, sys

REPO_PATH = "/kaggle/working/amorphous-yolo"
if not os.path.exists(f"{REPO_PATH}/.git"):
    print("Cloning amorphous-yolo...")
    os.system(f"git clone https://github.com/nipun-taneja/amorphous-yolo.git {REPO_PATH}")
else:
    print("Repo already present — skipping clone.")

if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

os.chdir(REPO_PATH)
print(f"Working directory: {os.getcwd()}")
""", "colondb-gitclone"))

# ── Cell 5: Constants ──────────────────────────────────────────────────────────
cells.append(code(
"""# --- All experiment constants (single source of truth for this notebook)
import math, time, re
from pathlib import Path
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR   = Path("/kaggle/working/amorphous-yolo")
DATASET_ROOT  = PROJECT_DIR / "datasets" / "cvc-colondb"
EXPERIMENTS   = PROJECT_DIR / "experiments_colondb"
ANALYSIS_DIR  = EXPERIMENTS / "analysis"
MANIFEST_PATH = EXPERIMENTS / "manifest.json"
EXPERIMENTS.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

KAGGLE_OUTPUT   = Path("/kaggle/working")
DRIVE_AVAILABLE = False  # no Drive on Kaggle

# ── Training hyper-parameters ─────────────────────────────────────────────────
EPOCHS   = 20
IMGSZ    = 640
DEVICE   = 0
MODEL_PT = "yolo26n.pt"
WANDB_PROJECT = "cvc-colondb-aeiou"

# ── Seeds ─────────────────────────────────────────────────────────────────────
SEEDS = [42]

# ── Loss keys ─────────────────────────────────────────────────────────────────
BASELINE_LOSS_NAMES = ["iou", "giou", "diou", "ciou", "eiou", "eciou", "siou", "wiou"]
AEIOU_RIGIDITIES    = [round(x * 0.1, 1) for x in range(1, 11)]

def _fmt_r(r):
    return str(r).replace(".", "p")

ALL_LOSS_KEYS = BASELINE_LOSS_NAMES + [f"aeiou_r{_fmt_r(r)}" for r in AEIOU_RIGIDITIES]

# ── SPLIT_CONFIGS populated after YAML creation cell ─────────────────────────
SPLIT_CONFIGS = {}

# ── Visual palette ────────────────────────────────────────────────────────────
PALETTE = {
    "iou": "#888888", "giou": "#BC6C25", "diou": "#606C38", "ciou": "#DDA15E",
    "eiou": "#E63946", "eciou": "#9B2226", "siou": "#6A0572", "wiou": "#FF6B6B",
    "aeiou_r0p1": "#023E8A", "aeiou_r0p2": "#0077B6", "aeiou_r0p3": "#00B4D8",
    "aeiou_r0p4": "#48CAE4", "aeiou_r0p5": "#90E0EF", "aeiou_r0p6": "#2A9D8F",
    "aeiou_r0p7": "#52B788", "aeiou_r0p8": "#74C69D", "aeiou_r0p9": "#95D5B2",
    "aeiou_r1p0": "#6A4C93",
}

LOSS_LABELS = {
    "iou": "IoU", "giou": "GIoU", "diou": "DIoU", "ciou": "CIoU",
    "eiou": "EIoU", "eciou": "ECIoU", "siou": "SIoU", "wiou": "WIoU",
}
for r in AEIOU_RIGIDITIES:
    LOSS_LABELS[f"aeiou_r{_fmt_r(r)}"] = f"AEIoU lam={r}"

n_total = (len(BASELINE_LOSS_NAMES) + len(AEIOU_RIGIDITIES)) * 3 * len(SEEDS)
print("Constants loaded.")
print(f"  Baselines : {BASELINE_LOSS_NAMES}")
print(f"  AEIoU grid: {AEIOU_RIGIDITIES}")
print(f"  Seeds     : {SEEDS}")
print(f"  Total planned runs: {n_total}")
""", "colondb-constants"))

# ── Cell 6: Section 2 markdown ─────────────────────────────────────────────────
cells.append(md(
"""## Section 2 · CVC-ColonDB Dataset

### Why CVC-ColonDB for AEIoU validation?

CVC-ColonDB (Tajbakhsh et al., 2015) contains **380 colonoscopy frames** with
pixel-level polyp segmentation masks. It is used as a cross-dataset test in
many polyp detection papers, providing a complementary view to CVC-ClinicDB.

| Property | Value |
|---|---|
| Total images | 380 |
| Train / Val  | ~304 / ~76 (80/20 random split, seed 42) |
| Classes | 1 (polyp) |
| Image size | variable |
| Mask format | binary PNG |
| Noise splits | clean, low (σ=0.02), high (σ=0.08) |
""", "colondb-sec2-md"))

# ── Cell 7: WandB setup ────────────────────────────────────────────────────────
cells.append(code(
"""# --- WandB setup via Kaggle secrets (optional — silently skipped if missing)
import os, wandb

WANDB_PROJECT = "cvc-colondb-aeiou"
try:
    from kaggle_secrets import UserSecretsClient
    wandb_key = UserSecretsClient().get_secret("WANDB_API_KEY")
    wandb.login(key=wandb_key, quiet=True)
    print(f"WandB logged in. Project: {WANDB_PROJECT}")
except Exception as e:
    print(f"WandB not configured ({e}) — training will run without tracking.")
    os.environ["WANDB_DISABLED"] = "true"
""", "colondb-wandb"))

# ── Cell 8: Kaggle output check ────────────────────────────────────────────────
cells.append(code(
"""# --- Check output directory and list any previously completed runs
from pathlib import Path
import pandas as pd

print(f"Experiment dir: {EXPERIMENTS}")
existing = sorted(EXPERIMENTS.glob("*/results.csv"))
if existing:
    print(f"Previously completed runs: {len(existing)}")
    for p in existing[:10]:
        df = pd.read_csv(p)
        print(f"  {p.parent.name}: {len(df)} epochs")
    if len(existing) > 10:
        print(f"  ... and {len(existing)-10} more")
else:
    print("No previous runs found. Starting fresh.")
""", "colondb-output-check"))

# ── Cell 9: Dataset setup ──────────────────────────────────────────────────────
cells.append(code(
"""# --- Setup CVC-ColonDB from Kaggle input dataset
# Dataset: hannibal93/cvc-colondb on Kaggle
# Attach via: Add Data -> Search "cvc-colondb" (hannibal93) -> Add
# Expected Kaggle path: /kaggle/input/cvc-colondb/CVC-ColonDB/images/ + masks/
import shutil
from pathlib import Path

def _find_images(d, exts=("*.png", "*.jpg", "*.bmp")):
    \"\"\"Return sorted list of all image files in directory d.\"\"\"
    imgs = []
    for ext in exts:
        imgs.extend(Path(d).glob(ext))
    return sorted(set(imgs), key=lambda p: p.stem)

# ── Recursive scan: find any 'images' dir under /kaggle/input with >=350 PNGs ─
# Handles any nesting level:
#   /kaggle/input/cvc-colondb/images/          (flat)
#   /kaggle/input/cvc-colondb/CVC-ColonDB/images/  (one subdir)
# Print all input dirs found so user can debug if needed
print("Scanning /kaggle/input/ for CVC-ColonDB images...")
import os
for root, dirs, files in os.walk("/kaggle/input"):
    print(f"  dir: {root}  ({len(files)} files)")

orig_dir = None
mask_dir = None

# Find images/ directory with >=350 images
for img_dir in sorted(Path("/kaggle/input").rglob("images")):
    if not img_dir.is_dir():
        continue
    imgs = _find_images(img_dir)
    if len(imgs) >= 350:
        orig_dir = img_dir
        print(f"\\nFound images: {img_dir} ({len(imgs)} files)")
        break

# Find sibling masks/ directory (same parent as images/)
if orig_dir is not None:
    parent = orig_dir.parent
    for mask_name in ["masks", "Masks", "Ground Truth", "GroundTruth", "ground_truth"]:
        cand = parent / mask_name
        if cand.is_dir():
            msks = _find_images(cand)
            if len(msks) >= 350:
                mask_dir = cand
                print(f"Found masks : {cand} ({len(msks)} files)")
                break

assert orig_dir is not None, (
    "Could not find CVC-ColonDB images (>=350 PNG files) under /kaggle/input/. "
    "Attach the dataset 'hannibal93/cvc-colondb' via Add Data."
)
assert mask_dir is not None, (
    f"Found images at {orig_dir} but no masks/ sibling with >=350 files. "
    "Check that your dataset zip contains both images/ and masks/ folders."
)

# ── Copy to DATASET_ROOT (idempotent) ─────────────────────────────────────────
(DATASET_ROOT / "images").mkdir(parents=True, exist_ok=True)
(DATASET_ROOT / "masks").mkdir(parents=True, exist_ok=True)

n_existing_imgs = len(_find_images(DATASET_ROOT / "images"))
if n_existing_imgs >= 350:
    print(f"Dataset already at DATASET_ROOT ({n_existing_imgs} images) — skipping copy.")
else:
    print("Copying images and masks to DATASET_ROOT...")
    for img_path in _find_images(orig_dir):
        dst = DATASET_ROOT / "images" / img_path.name
        if not dst.exists():
            shutil.copy2(img_path, dst)
    for msk_path in _find_images(mask_dir):
        dst = DATASET_ROOT / "masks" / msk_path.name
        if not dst.exists():
            shutil.copy2(msk_path, dst)

n_imgs  = len(_find_images(DATASET_ROOT / "images"))
n_masks = len(_find_images(DATASET_ROOT / "masks"))
print(f"DATASET_ROOT: {DATASET_ROOT}")
print(f"  Images: {n_imgs}  |  Masks: {n_masks}")
assert n_imgs == n_masks >= 350, f"Expected >=350 matched pairs, got {n_imgs}/{n_masks}"
print("Dataset setup complete.")
""", "colondb-dataset-setup"))

# ── Cell 10: Convert masks -> YOLO labels + train/val split ───────────────────
cells.append(code(
"""# --- Convert CVC-ColonDB masks -> YOLO bbox labels + 80/20 train/val split
# CVC-ColonDB provides binary PNG masks (white=polyp, black=background).
import cv2, numpy as np, random, shutil
from pathlib import Path

TRAIN_DIR = DATASET_ROOT / "train"
VALID_DIR = DATASET_ROOT / "valid"

def _find_images(d, exts=("*.png", "*.jpg", "*.bmp")):
    imgs = []
    for ext in exts:
        imgs.extend(Path(d).glob(ext))
    return sorted(set(imgs), key=lambda p: p.stem)

def _mask_to_yolo_bbox(mask_path):
    \"\"\"Load binary mask and return normalised (cx, cy, w, h) or None.\"\"\"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    _, binary = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    all_pts = np.vstack(contours)
    x, y, w, h = cv2.boundingRect(all_pts)
    H, W = mask.shape
    cx = float(np.clip((x + w / 2) / W, 0.0, 1.0))
    cy = float(np.clip((y + h / 2) / H, 0.0, 1.0))
    bw = float(np.clip(w / W, 0.01, 1.0))
    bh = float(np.clip(h / H, 0.01, 1.0))
    return cx, cy, bw, bh

# Check idempotency
n_train_check = len(_find_images(TRAIN_DIR / "images")) if (TRAIN_DIR / "images").exists() else 0
if n_train_check >= 280:
    print(f"Train/val split already exists ({n_train_check} train images) — skipping.")
else:
    print("Converting masks -> YOLO labels and splitting 80/20 (seed=42)...")
    for split_dir in [TRAIN_DIR, VALID_DIR]:
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)

    all_imgs = _find_images(DATASET_ROOT / "images")
    random.seed(42)
    random.shuffle(all_imgs)
    split_idx  = int(0.8 * len(all_imgs))
    splits_map = {"train": all_imgs[:split_idx], "valid": all_imgs[split_idx:]}

    skipped = 0
    for split_name, img_list in splits_map.items():
        out_img_dir = DATASET_ROOT / split_name / "images"
        out_lbl_dir = DATASET_ROOT / split_name / "labels"
        for img_path in img_list:
            stem = img_path.stem
            mask_path = None
            for mext in [".png", ".jpg", ".bmp"]:
                cand = DATASET_ROOT / "masks" / f"{stem}{mext}"
                if cand.exists():
                    mask_path = cand
                    break
            if mask_path is None:
                skipped += 1
                continue
            bbox = _mask_to_yolo_bbox(mask_path)
            if bbox is None:
                skipped += 1
                continue
            cx, cy, bw, bh = bbox
            shutil.copy2(img_path, out_img_dir / img_path.name)
            (out_lbl_dir / f"{stem}.txt").write_text(
                f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\\n"
            )

    n_tr = len(_find_images(TRAIN_DIR / "images"))
    n_vl = len(_find_images(VALID_DIR / "images"))
    print(f"  Skipped {skipped} (blank/missing mask)")
    print(f"  Train: {n_tr} images  |  Valid: {n_vl} images")

print("Mask conversion complete.")
""", "colondb-mask-convert"))

# ── Cell 11: Noise splits ──────────────────────────────────────────────────────
cells.append(code(
"""# --- Build noise-perturbed validation splits (idempotent)
import numpy as np, shutil
from pathlib import Path

SIGMA_LOW  = 0.02
SIGMA_HIGH = 0.08

def _find_images(d, exts=("*.png", "*.jpg", "*.bmp")):
    imgs = []
    for ext in exts:
        imgs.extend(Path(d).glob(ext))
    return sorted(set(imgs), key=lambda p: p.stem)

def _jitter_label_file(src_lbl, dst_lbl, sigma, rng):
    \"\"\"Perturb all YOLO bbox coords in a label file by N(0, sigma).\"\"\"
    lines = Path(src_lbl).read_text().strip().split("\\n")
    out = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        cls_id = parts[0]
        cx, cy, w, h = [float(v) for v in parts[1:5]]
        cx = float(np.clip(cx + rng.normal(0, sigma), 0.0, 1.0))
        cy = float(np.clip(cy + rng.normal(0, sigma), 0.0, 1.0))
        w  = float(np.clip(w  + rng.normal(0, sigma), 0.01, 1.0))
        h  = float(np.clip(h  + rng.normal(0, sigma), 0.01, 1.0))
        out.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    Path(dst_lbl).write_text("\\n".join(out) + "\\n")


def build_noise_splits(root, seed=42):
    src_img_dir = root / "valid" / "images"
    src_lbl_dir = root / "valid" / "labels"
    rng = np.random.default_rng(seed)

    for split_name, sigma in [("valid_low", SIGMA_LOW), ("valid_high", SIGMA_HIGH)]:
        dst_img_dir = root / split_name / "images"
        dst_lbl_dir = root / split_name / "labels"
        n_existing  = len(_find_images(dst_img_dir)) if dst_img_dir.exists() else 0

        if n_existing >= 60:
            print(f"  {split_name}: already exists ({n_existing} images) — skipping.")
            continue

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in _find_images(src_img_dir):
            stem    = img_path.stem
            dst_img = dst_img_dir / img_path.name
            if not dst_img.exists():
                try:
                    dst_img.symlink_to(img_path.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy2(img_path, dst_img)
            src_lbl = src_lbl_dir / f"{stem}.txt"
            dst_lbl = dst_lbl_dir / f"{stem}.txt"
            if src_lbl.exists():
                _jitter_label_file(src_lbl, dst_lbl, sigma, rng)

        n = len(_find_images(dst_img_dir))
        print(f"  {split_name}: {n} images (sigma={sigma})")


build_noise_splits(DATASET_ROOT)
print("Noise splits ready.")
""", "colondb-noise-splits"))

# ── Cell 12: Create YAML files ─────────────────────────────────────────────────
cells.append(code(
"""# --- Create YOLO dataset YAML files at runtime (runtime paths, no Colab hardcoding)
from pathlib import Path

DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

yaml_defs = {
    "cvc_colondb.yaml":      "valid/images",
    "cvc_colondb_low.yaml":  "valid_low/images",
    "cvc_colondb_high.yaml": "valid_high/images",
}

for fname, val_split in yaml_defs.items():
    content = (
        f"path: {DATASET_ROOT}\\n"
        f"train: train/images\\n"
        f"val: {val_split}\\n"
        f"nc: 1\\n"
        f"names: ['polyp']\\n"
    )
    yaml_path = DATA_DIR / fname
    yaml_path.write_text(content)
    print(f"Created: {yaml_path}")

SPLIT_CONFIGS = {
    "clean": DATA_DIR / "cvc_colondb.yaml",
    "low":   DATA_DIR / "cvc_colondb_low.yaml",
    "high":  DATA_DIR / "cvc_colondb_high.yaml",
}
print(f"\\nSPLIT_CONFIGS:")
for k, v in SPLIT_CONFIGS.items():
    print(f"  {k}: {v}")
""", "colondb-yaml-create"))

# ── Cell 13: Verify splits ─────────────────────────────────────────────────────
cells.append(code(
"""# --- Verify all three split configs and count images
import yaml
from pathlib import Path

def _find_images(d, exts=("*.png", "*.jpg", "*.bmp")):
    imgs = []
    for ext in exts:
        imgs.extend(Path(d).glob(ext))
    return sorted(set(imgs), key=lambda p: p.stem)

print(f"{'Split':<10} {'YAML':<35} {'Val images':>12}")
print("-" * 60)
for split_name, cfg_path in SPLIT_CONFIGS.items():
    assert cfg_path.exists(), f"Missing yaml: {cfg_path}"
    cfg = yaml.safe_load(cfg_path.read_text())
    val_dir = DATASET_ROOT / cfg["val"]
    n_imgs  = len(_find_images(val_dir))
    status  = "OK" if n_imgs >= 60 else "CHECK"
    print(f"{split_name:<10} {cfg_path.name:<35} {n_imgs:>10}  {status}")

n_train = len(_find_images(TRAIN_DIR / "images"))
print(f"\\nTrain images (shared): {n_train}")
assert n_train >= 280, f"Expected >=280 train images, got {n_train}"
print("All splits verified.")
""", "colondb-verify"))

# ── Cell 14: Section 3 markdown ────────────────────────────────────────────────
cells.append(md(
"""## Section 3 · Monkey-Patch Infrastructure

BboxLoss.forward in Ultralytics 8.4.9 is replaced at runtime to inject custom
loss functions. The original forward is saved once and always restored after
each training run via `restore_loss()`.
""", "colondb-sec3-md"))

# ── Cell 15: Monkey-patch ──────────────────────────────────────────────────────
cells.append(code(
"""# --- Full monkey-patch implementation
import types, torch, torch.nn.functional as F
import ultralytics.utils.loss as loss_mod

_ORIGINAL_BBOX_FORWARD = loss_mod.BboxLoss.forward


def _make_bbox_forward(loss_fn_instance):
    def bbox_loss_forward(
        self, pred_dist, pred_bboxes, anchor_points,
        target_bboxes, target_scores, target_scores_sum,
        fg_mask, imgsz, stride,
    ):
        weight   = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        per_box  = loss_fn_instance(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = (per_box.unsqueeze(-1) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = loss_mod.bbox2dist(
                anchor_points, target_bboxes, self.dfl_loss.reg_max - 1
            )
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[fg_mask],
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = loss_mod.bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist_s = pred_dist * stride
            pred_dist_s[..., 0::2] /= imgsz[1]
            pred_dist_s[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist_s[fg_mask], target_ltrb[fg_mask],
                          reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl

    return bbox_loss_forward


def patch_loss(loss_fn_instance):
    loss_mod.BboxLoss.forward = _make_bbox_forward(loss_fn_instance)
    rig = getattr(loss_fn_instance, "rigidity", None)
    tag = f"(lam={rig})" if rig is not None else ""
    print(f"  [PATCH] BboxLoss.forward -> {type(loss_fn_instance).__name__}{tag}")


def restore_loss():
    loss_mod.BboxLoss.forward = _ORIGINAL_BBOX_FORWARD


print("Patch infrastructure ready.")
""", "colondb-monkeypatch"))

# ── Cell 16: Patch verify ──────────────────────────────────────────────────────
cells.append(code(
"""# --- Verify patch round-trip
from src.losses import EIoULoss
patch_loss(EIoULoss(reduction="none"))
restore_loss()
assert loss_mod.BboxLoss.forward is _ORIGINAL_BBOX_FORWARD
print("Patch round-trip OK.")
""", "colondb-patch-verify"))

# ── Cell 17: Loss registry ─────────────────────────────────────────────────────
cells.append(code(
"""# --- Loss registry: all baselines + AEIoU grid
from src.losses import (IoULoss, GIoULoss, DIoULoss, CIoULoss,
                        EIoULoss, ECIoULoss, SIoULoss, WIoULoss, AEIoULoss)

BASELINE_LOSS_REGISTRY = {
    "iou":   IoULoss(reduction="none"),
    "giou":  GIoULoss(reduction="none"),
    "diou":  DIoULoss(reduction="none"),
    "ciou":  CIoULoss(reduction="none"),
    "eiou":  EIoULoss(reduction="none"),
    "eciou": ECIoULoss(reduction="none"),
    "siou":  SIoULoss(reduction="none"),
    "wiou":  WIoULoss(reduction="none"),
}

AEIOU_LOSS_REGISTRY = {
    f"aeiou_r{_fmt_r(r)}": AEIoULoss(rigidity=r, reduction="none")
    for r in AEIOU_RIGIDITIES
}

ALL_LOSS_REGISTRY = {**BASELINE_LOSS_REGISTRY, **AEIOU_LOSS_REGISTRY}

n_total = len(ALL_LOSS_REGISTRY) * len(SPLIT_CONFIGS) * len(SEEDS)
print(f"Baselines ({len(BASELINE_LOSS_REGISTRY)}): {list(BASELINE_LOSS_REGISTRY)}")
print(f"AEIoU grid ({len(AEIOU_LOSS_REGISTRY)}): lambda = {AEIOU_RIGIDITIES}")
print(f"Total planned runs: {n_total}")
""", "colondb-loss-registry"))

# ── Cell 18: Training functions ────────────────────────────────────────────────
cells.append(code(
"""# --- Training functions: run_training + helpers
import json, shutil
from pathlib import Path as _Path
from datetime import datetime
from ultralytics import YOLO


def _load_manifest():
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def write_manifest_entry(run_name, meta):
    manifest = _load_manifest()
    manifest[run_name] = meta
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def sync_to_drive(run_name):
    pass  # No Drive on Kaggle


def make_epoch_checkpoint_callback(run_name):
    def _on_epoch_end(trainer):
        pass
    return _on_epoch_end


def run_training(loss_name, loss_fn, split_name, yaml_path,
                 seed=42, epochs=None, imgsz=None, device=None):
    \"\"\"Train one YOLO26n model. Idempotent — skips completed runs.\"\"\"
    epochs = epochs if epochs is not None else EPOCHS
    imgsz  = imgsz  if imgsz  is not None else IMGSZ
    device = device if device is not None else DEVICE

    # ColonDB run name prefix
    run_name = f"colondb_yolo26n_{loss_name}_{split_name}_s{seed}_e{epochs}"
    run_dir  = EXPERIMENTS / run_name

    # ── Skip if already completed ──────────────────────────────────────────────
    csv_path = run_dir / "results.csv"
    if csv_path.exists():
        import pandas as _pd
        n_rows = len(_pd.read_csv(csv_path))
        if n_rows >= epochs:
            print(f"[SKIP] {run_name} ({n_rows} epochs complete)")
            return run_dir

    local_last_pt = run_dir / "weights" / "last.pt"
    resuming = local_last_pt.exists() and not (run_dir / "results.csv").exists()

    print(f"\\n{'='*68}")
    if resuming:
        print(f"[RESUME] {run_name}")
    else:
        print(f"[START ] {run_name}")
        print(f"  loss={loss_name}  split={split_name}  seed={seed}  epochs={epochs}")
    print(f"{'='*68}")

    meta = {
        "loss": loss_name, "split": split_name, "seed": seed,
        "epochs": epochs,
        "rigidity": float(getattr(loss_fn, "rigidity", -1) or -1),
        "run_dir": str(run_dir),
        "timestamp": datetime.now().isoformat(),
        "status": "running", "resumed": resuming,
    }
    write_manifest_entry(run_name, meta)

    t_start = time.time()
    try:
        import os as _os
        _os.environ["WANDB_PROJECT"] = WANDB_PROJECT
        _os.environ["WANDB_NAME"]    = run_name
        _os.environ["WANDB_TAGS"]    = f"{loss_name},{split_name}"

        patch_loss(loss_fn)

        if resuming:
            model = YOLO(str(local_last_pt))
            model.add_callback("on_train_epoch_end",
                               make_epoch_checkpoint_callback(run_name))
            results = model.train(resume=True)
        else:
            model = YOLO(MODEL_PT)
            model.add_callback("on_train_epoch_end",
                               make_epoch_checkpoint_callback(run_name))
            results = model.train(
                data=str(yaml_path),
                epochs=epochs,
                imgsz=imgsz,
                project=str(EXPERIMENTS),
                name=run_name,
                device=device,
                seed=seed,
                exist_ok=True,
            )

        try:
            (run_dir / "run_meta.json").write_text(
                json.dumps(results.results_dict, indent=2)
            )
        except Exception as e:
            print(f"  [WARN] run_meta.json: {e}")

        meta["status"]      = "complete"
        meta["elapsed_sec"] = round(time.time() - t_start, 1)

        try:
            import wandb as _wandb
            if _wandb.run is not None:
                _wandb.finish()
        except Exception:
            pass

    except Exception as e:
        print(f"  [ERROR] {run_name}: {e}")
        meta["status"] = "failed"
        meta["error"]  = str(e)
        raise

    finally:
        restore_loss()
        write_manifest_entry(run_name, meta)

    print(f"[DONE] {run_name}  ({meta.get('elapsed_sec', 0):.0f}s)")
    return run_dir


print("run_training() ready.")
print(f"Manifest: {MANIFEST_PATH}")
""", "colondb-training-fn"))

# ── Cell 19: Section 7 markdown ────────────────────────────────────────────────
cells.append(md(
"""## Section 7 · Baseline Training

Training all 8 standard IoU-family losses × 3 splits × 1 seed = **24 runs**.

Estimated time: ~1.2 hours on T4 GPU (fewer images → ~3 min/run × 24).
""", "colondb-sec7-md"))

# ── Cell 20: Baseline training ─────────────────────────────────────────────────
cells.append(code(
"""# --- Baseline training: all 8 standard losses x 3 splits x N seeds
print(f"Starting baseline training: {len(BASELINE_LOSS_REGISTRY)} losses x "
      f"{len(SPLIT_CONFIGS)} splits x {len(SEEDS)} seed(s)")
print(f"Estimated time: ~{len(BASELINE_LOSS_REGISTRY)*len(SPLIT_CONFIGS)*len(SEEDS)*3.0/60:.1f} hours\\n")

for loss_name, loss_fn in BASELINE_LOSS_REGISTRY.items():
    for split_name, cfg_path in SPLIT_CONFIGS.items():
        for seed in SEEDS:
            run_training(
                loss_name=loss_name, loss_fn=loss_fn,
                split_name=split_name, yaml_path=cfg_path, seed=seed,
            )

restore_loss()
n_base = len(BASELINE_LOSS_REGISTRY) * len(SPLIT_CONFIGS) * len(SEEDS)
print(f"\\nAll {n_base} baseline runs complete (or skipped).")
""", "colondb-baseline-train"))

# ── Cell 21: Section 8 markdown ────────────────────────────────────────────────
cells.append(md(
"""## Section 8 · AEIoU Rigidity Grid Search

Sweeping lambda from 0.1 → 1.0 in steps of 0.1 across all 3 noise splits.

10 lambdas × 3 splits × 1 seed = **30 runs**.
Estimated time: ~1.5 hours on T4 GPU.
""", "colondb-sec8-md"))

# ── Cell 22: AEIoU grid training ───────────────────────────────────────────────
cells.append(code(
"""# --- AEIoU rigidity grid training (10 lambdas x 3 splits x N seeds)
total = len(AEIOU_RIGIDITIES) * len(SPLIT_CONFIGS) * len(SEEDS)
done  = 0
print(f"Starting AEIoU grid: {total} runs total\\n")

for r in AEIOU_RIGIDITIES:
    loss_name = f"aeiou_r{_fmt_r(r)}"
    loss_fn   = AEIOU_LOSS_REGISTRY[loss_name]
    for split_name, cfg_path in SPLIT_CONFIGS.items():
        for seed in SEEDS:
            done += 1
            print(f"[{done}/{total}] lam={r}  split={split_name}  seed={seed}")
            run_training(
                loss_name=loss_name, loss_fn=loss_fn,
                split_name=split_name, yaml_path=cfg_path, seed=seed,
            )

restore_loss()
print(f"\\nAll {total} AEIoU grid runs complete (or skipped).")
""", "colondb-aeiou-train"))

# ── Cell 23: Section 9 markdown ────────────────────────────────────────────────
cells.append(md("## Section 9 · Results Collection\n", "colondb-sec9-md"))

# ── Cell 24: Load results ──────────────────────────────────────────────────────
cells.append(code(
"""# --- Load all results.csv files into a single flat DataFrame
import pandas as pd

CACHE_CSV = EXPERIMENTS / "all_results_combined.csv"

def load_all_results(force_rebuild=False):
    if CACHE_CSV.exists() and not force_rebuild:
        print(f"Loading from cache: {CACHE_CSV}")
        return pd.read_csv(CACHE_CSV)

    print("Building combined results from individual CSVs...")
    dfs = []
    for loss_name in ALL_LOSS_KEYS:
        for split_name in SPLIT_CONFIGS:
            for seed in SEEDS:
                run_name = f"colondb_yolo26n_{loss_name}_{split_name}_s{seed}_e{EPOCHS}"
                csv_path = EXPERIMENTS / run_name / "results.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    df.columns = df.columns.str.strip()
                    df["run_name"] = run_name
                    df["loss"]     = loss_name
                    df["split"]    = split_name
                    df["seed"]     = seed
                    df["rigidity"] = (
                        float(loss_name.split("_r")[1].replace("p", "."))
                        if "aeiou" in loss_name else -1.0
                    )
                    dfs.append(df)
                else:
                    print(f"  [MISSING] {csv_path}")

    if not dfs:
        raise RuntimeError("No results found. Run training cells first.")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(CACHE_CSV, index=False)
    print(f"Cached {len(dfs)} runs -> {CACHE_CSV}")
    return df_all

df_all = load_all_results(force_rebuild=True)
print(f"\\nDataFrame shape: {df_all.shape}")
print(f"Runs found: {df_all['run_name'].nunique()} / {len(ALL_LOSS_KEYS)*len(SPLIT_CONFIGS)*len(SEEDS)}")
""", "colondb-load-results"))

# ── Cell 25: Section 10 markdown ───────────────────────────────────────────────
cells.append(md("## Section 10 · Summary Table\n", "colondb-sec10-md"))

# ── Cell 26: Summary table ─────────────────────────────────────────────────────
cells.append(code(
"""# --- Build master pivot table
import pandas as pd, numpy as np

MAP50_COL = "metrics/mAP50(B)"
MAP95_COL = "metrics/mAP50-95(B)"

df_final = (
    df_all.sort_values("epoch")
          .groupby("run_name").last().reset_index()
)

agg_rows = []
for loss_name in ALL_LOSS_KEYS:
    for split in ["clean", "low", "high"]:
        sub = df_final[(df_final["loss"] == loss_name) & (df_final["split"] == split)]
        if sub.empty:
            continue
        row = {"loss": loss_name, "split": split,
               "label": LOSS_LABELS.get(loss_name, loss_name)}
        for col, tag in [(MAP95_COL, "map95"), (MAP50_COL, "map50")]:
            if col in sub.columns:
                row[f"{tag}_mean"] = sub[col].mean()
                row[f"{tag}_std"]  = sub[col].std() if len(sub) > 1 else 0.0
        agg_rows.append(row)

df_agg = pd.DataFrame(agg_rows)

pivot95 = df_agg.pivot_table(index="loss", columns="split", values="map95_mean")
pivot95 = pivot95[["clean", "low", "high"]]
pivot95["robust_ratio"] = pivot95["high"] / pivot95["clean"].replace(0, float("nan"))
pivot95 = pivot95.sort_values("clean", ascending=False)

pivot50 = df_agg.pivot_table(index="loss", columns="split", values="map50_mean")
pivot50 = pivot50[["clean", "low", "high"]]

pivot95.to_csv(ANALYSIS_DIR / "summary_map95.csv")
pivot50.to_csv(ANALYSIS_DIR / "summary_map50.csv")

print("=== mAP50-95 by Loss and Split ===")
print(pivot95.round(4).to_string())
print("\\n=== mAP50 by Loss and Split ===")
print(pivot50.round(4).to_string())
""", "colondb-summary-table"))

# ── Cell 27: Section 11 markdown ───────────────────────────────────────────────
cells.append(md("## Section 11 · Analysis Figures\n", "colondb-sec11-md"))

# ── Cell 28: Fig 1 bar chart ───────────────────────────────────────────────────
cells.append(code(
"""# --- Fig 1: Final mAP50-95 bar chart — ALL losses grouped by split
import matplotlib.pyplot as plt, numpy as np

x_labels       = BASELINE_LOSS_NAMES + [f"aeiou_r{_fmt_r(r)}" for r in AEIOU_RIGIDITIES]
display_labels = [LOSS_LABELS.get(l, l) for l in x_labels]
split_colors   = {"clean": "#4CAF50", "low": "#FFC107", "high": "#F44336"}
x_pos, width   = np.arange(len(x_labels)), 0.25

fig, ax = plt.subplots(figsize=(18, 6))
for i, split in enumerate(["clean", "low", "high"]):
    vals = [pivot95.loc[l, split] if l in pivot95.index else 0 for l in x_labels]
    ax.bar(x_pos + i * width, vals, width=width,
           color=split_colors[split], alpha=0.85, label=split)

ax.axvline(x=len(BASELINE_LOSS_NAMES) - 0.5, color="black", linestyle=":", lw=1.5, alpha=0.4)
ax.set_xticks(x_pos + width)
ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("mAP@[.5:.95]", fontsize=11)
ax.set_title("CVC-ColonDB — Final mAP50-95: All Losses x 3 Splits", fontsize=13, fontweight="bold")
ax.legend(title="Split", fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "01_final_map95_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved -> 01_final_map95_comparison.png")
""", "colondb-fig1"))

# ── Cell 29: Lambda curve ──────────────────────────────────────────────────────
cells.append(code(
"""# --- Fig 2: Lambda-vs-mAP curve with baseline reference lines (KEY FIGURE)
import matplotlib.pyplot as plt, numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("CVC-ColonDB — AEIoU Rigidity Sweep vs Baselines", fontsize=13, fontweight="bold")

for ax, metric_col, ylabel in [
    (axes[0], "map95_mean", "mAP50-95"),
    (axes[1], "map50_mean", "mAP50"),
]:
    for split, ls, marker in [("clean", "-", "o"), ("low", "--", "s"), ("high", ":", "^")]:
        lambdas, maps = [], []
        for r in AEIOU_RIGIDITIES:
            lname = f"aeiou_r{_fmt_r(r)}"
            row = df_agg[(df_agg["loss"] == lname) & (df_agg["split"] == split)]
            if not row.empty:
                lambdas.append(r)
                maps.append(row[metric_col].values[0])
        if lambdas:
            lbl = f"AEIoU ({split})"
            ax.plot(lambdas, maps, ls + marker, color="#0077B6", lw=2,
                    markersize=5, label=lbl)

    for bname in BASELINE_LOSS_NAMES:
        row = df_agg[(df_agg["loss"] == bname) & (df_agg["split"] == "clean")]
        if not row.empty:
            val = row[metric_col].values[0]
            ax.axhline(y=val, color=PALETTE.get(bname, "#888"),
                       linestyle="--", lw=1.0, alpha=0.7,
                       label=f"{LOSS_LABELS.get(bname, bname)} ({val:.3f})")

    ax.set_xlabel("AEIoU lambda (rigidity)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "02_lambda_vs_map.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved -> 02_lambda_vs_map.png")
""", "colondb-fig2"))

# ── Cell 30: Learning curves ───────────────────────────────────────────────────
cells.append(code(
"""# --- Fig 3: Learning curves — train loss + val mAP (clean split)
import matplotlib.pyplot as plt

MAP95_COL = "metrics/mAP50-95(B)"
LOSS_COL  = "train/box_loss"
vis_losses = BASELINE_LOSS_NAMES + ["aeiou_r0p1", "aeiou_r0p3", "aeiou_r0p5", "aeiou_r1p0"]

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle("CVC-ColonDB — Learning Curves (clean split)", fontsize=13, fontweight="bold")

for loss_name in vis_losses:
    color = PALETTE.get(loss_name, "#888")
    label = LOSS_LABELS.get(loss_name, loss_name)
    matching = df_all[(df_all["loss"] == loss_name) & (df_all["split"] == "clean")]
    if matching.empty:
        continue
    avg = matching.groupby("epoch").mean(numeric_only=True).reset_index()
    lw = 2.2 if loss_name in ["eiou", "eciou"] or "aeiou" in loss_name else 1.2
    ls = "-" if loss_name in BASELINE_LOSS_NAMES else "--"
    if LOSS_COL in avg.columns:
        axes[0].plot(avg["epoch"], avg[LOSS_COL], color=color, lw=lw, ls=ls, label=label)
    if MAP95_COL in avg.columns:
        axes[1].plot(avg["epoch"], avg[MAP95_COL], color=color, lw=lw, ls=ls, label=label)

for ax in axes:
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("Training box loss")
axes[1].set_ylabel("Val mAP50-95")
axes[1].set_xlabel("Epoch")
plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "03_learning_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved -> 03_learning_curves.png")
""", "colondb-fig3"))

# ── Cell 31: Noise robustness ──────────────────────────────────────────────────
cells.append(code(
"""# --- Fig 4: Noise robustness gap (mAP_clean - mAP_high)
import matplotlib.pyplot as plt, numpy as np

all_losses = BASELINE_LOSS_NAMES + [f"aeiou_r{_fmt_r(r)}" for r in AEIOU_RIGIDITIES]
gaps = []
for l in all_losses:
    if l in pivot95.index:
        gaps.append(float(pivot95.loc[l, "clean"]) - float(pivot95.loc[l, "high"]))
    else:
        gaps.append(0)

display_labels = [LOSS_LABELS.get(l, l) for l in all_losses]
bar_colors = [PALETTE.get(l, "#888") for l in all_losses]

fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(display_labels, gaps, color=bar_colors, edgecolor="white", lw=0.5)
if "eiou" in all_losses:
    eiou_gap = gaps[all_losses.index("eiou")]
    ax.axhline(y=eiou_gap, color="#E63946", linestyle="--", lw=1.5,
               label=f"EIoU gap = {eiou_gap:.4f}")
ax.set_ylabel("mAP gap (clean - high) — smaller = more robust")
ax.set_title("CVC-ColonDB — Noise Robustness Gap (All Losses)", fontsize=13, fontweight="bold")
ax.tick_params(axis="x", rotation=45)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "04_noise_robustness_gap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved -> 04_noise_robustness_gap.png")

import pandas as pd
ranking = []
for l in all_losses:
    if l not in pivot95.index:
        continue
    c, h = float(pivot95.loc[l, "clean"]), float(pivot95.loc[l, "high"])
    ranking.append({"loss": l, "label": LOSS_LABELS.get(l, l),
                    "clean": c, "high": h, "gap": c - h,
                    "ratio": h / c if c > 0 else 0})
rank_df = pd.DataFrame(ranking).sort_values("ratio", ascending=False).reset_index(drop=True)
rank_df.to_csv(ANALYSIS_DIR / "robustness_ranking.csv", index=False)
print("\\n=== Robustness Ranking (mAP_high/mAP_clean, higher=more robust) ===")
print(rank_df.to_string(index=False))
""", "colondb-fig4"))

# ── Cell 32: Save artifacts ────────────────────────────────────────────────────
cells.append(code(
"""# --- Save experiment artifacts summary
import json

print("=== CVC-ColonDB Experiment Artifact Summary ===\\n")
print(f"Experiments dir : {EXPERIMENTS}")
print(f"Analysis dir    : {ANALYSIS_DIR}")

if MANIFEST_PATH.exists():
    manifest = json.loads(MANIFEST_PATH.read_text())
    complete = sum(1 for v in manifest.values() if v.get("status") == "complete")
    failed   = sum(1 for v in manifest.values() if v.get("status") == "failed")
    print(f"\\nManifest: {len(manifest)} runs  |  {complete} complete  |  {failed} failed")

print("\\nAnalysis files:")
for f in sorted(ANALYSIS_DIR.glob("*")):
    print(f"  {f.name:<50} {f.stat().st_size/1024:.1f} KB")
""", "colondb-artifacts"))

# ── Cell 33: Section 14 header ────────────────────────────────────────────────
cells.append(md(
"""## Section 14 · Comprehensive Metrics Extraction

Computes and persists for every completed run:
- **COCO AP suite**: AP50, AP75, mAP50-95, APs/APm/APl, AR@1/10/100
- **PR curve**: precision[], recall[], F1[], AP per IoU threshold
- **Confusion matrix**: TP/FP/FN

Uses `DetectionValidator` directly with `get_cfg` to ensure `save_json=True` takes
effect and `jdict` is populated for pycocotools.

Image IDs derived with `hashlib.md5` consistently in Cell A (GT JSON) and Cell B
(predictions lookup) — same pattern proven in CVC-ClinicDB notebook.
""", "colondb-sec14-md"))

# ── Cell 34 (Cell A): COCO GT JSON ────────────────────────────────────────────
cells.append(code(
"""# --- Cell A: Build COCO-format ground-truth JSON from YOLO val labels (idempotent)
import cv2, json as _json, hashlib
from pathlib import Path

def _img_id(stem: str) -> int:
    \"\"\"Stable integer image ID. Uses int() for purely numeric stems
    (CVC-ColonDB: '1','2',...,'380'), else hashlib.md5. MUST match Cell B.
    \"\"\"
    if stem.isdigit():
        return int(stem)
    return int(hashlib.md5(stem.encode()).hexdigest(), 16) % (2 ** 31)


def _find_images(d, exts=("*.png", "*.jpg", "*.bmp")):
    imgs = []
    for ext in exts:
        imgs.extend(Path(d).glob(ext))
    return sorted(set(imgs), key=lambda p: p.stem)


def build_coco_gt_json(val_img_dir, val_lbl_dir, out_path, category_name="polyp"):
    \"\"\"Convert YOLO val labels -> COCO instances JSON for pycocotools.\"\"\"
    if Path(out_path).exists():
        print(f"COCO GT JSON already exists: {out_path}")
        return
    images, annotations, ann_id = [], [], 1
    for img_path in _find_images(val_img_dir):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        stem   = img_path.stem
        img_id = _img_id(stem)
        images.append({"id": img_id, "file_name": img_path.name, "width": W, "height": H})
        lbl = Path(val_lbl_dir) / f"{stem}.txt"
        if not lbl.exists():
            continue
        for line in lbl.read_text().strip().splitlines():
            if not line.strip():
                continue
            _, cx, cy, bw, bh = map(float, line.split())
            x1    = (cx - bw / 2) * W
            y1    = (cy - bh / 2) * H
            w_abs = bw * W
            h_abs = bh * H
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "bbox": [x1, y1, w_abs, h_abs],
                "area": w_abs * h_abs, "iscrowd": 0,
            })
            ann_id += 1
    coco = {
        "images":      images,
        "annotations": annotations,
        "categories":  [{"id": 1, "name": category_name}],
    }
    Path(out_path).write_text(_json.dumps(coco))
    print(f"COCO GT JSON: {out_path}  ({len(images)} imgs, {len(annotations)} anns)")


COCO_GT_JSON = DATASET_ROOT / "valid" / "coco_gt.json"
build_coco_gt_json(
    DATASET_ROOT / "valid" / "images",
    DATASET_ROOT / "valid" / "labels",
    COCO_GT_JSON,
)
""", "colondb-cell-a"))

# ── Cell 35 (Cell B): compute_and_persist_metrics ─────────────────────────────
cells.append(code(
"""# --- Cell B: compute_and_persist_metrics() using DetectionValidator
import hashlib, io, contextlib, json as _json
import numpy as np
from pathlib import Path
from ultralytics.cfg import get_cfg, DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionValidator

METRICS_DIR = EXPERIMENTS / "metrics"
METRICS_DIR.mkdir(exist_ok=True)


def compute_and_persist_metrics(run_name, weights_path, yaml_path,
                                coco_gt_json, force=False):
    out_dir     = METRICS_DIR / run_name
    done_marker = out_dir / "coco_ap_suite.json"
    if done_marker.exists() and not force:
        print(f"  [SKIP] {run_name}")
        return None
    if not Path(weights_path).exists():
        print(f"  [MISS] weights not found: {weights_path}")
        return None
    out_dir.mkdir(exist_ok=True)

    overrides = dict(
        model=str(weights_path), data=str(yaml_path),
        conf=0.001, iou=0.6, verbose=False,
        save_json=True, imgsz=640, split="val",
    )
    args      = get_cfg(DEFAULT_CFG, overrides)
    validator = DetectionValidator(args=args)
    validator()
    val_res = validator.metrics

    # ── 1. PR curve ──────────────────────────────────────────────────────────
    try:
        prec = np.atleast_2d(np.array(val_res.box.p))[0].tolist()
        rec  = np.atleast_2d(np.array(val_res.box.r))[0].tolist()
        f1   = np.atleast_2d(np.array(val_res.box.f1))[0].tolist()
    except Exception:
        prec, rec, f1 = [], [], []

    try:
        ap_per_iou = np.atleast_2d(val_res.box.ap)[0].tolist()
    except Exception:
        ap_per_iou = []

    pr_data = {
        "precision": prec, "recall": rec, "f1": f1,
        "ap50":    float(val_res.box.map50),
        "ap75":    float(val_res.box.map75),
        "map50_95": float(val_res.box.map),
        "ap_per_iou_threshold": ap_per_iou,
        "iou_thresholds": np.round(np.arange(0.5, 1.0, 0.05), 2).tolist(),
    }
    (out_dir / "pr_curve.json").write_text(_json.dumps(pr_data, indent=2))

    # ── 2. COCO AP suite ─────────────────────────────────────────────────────
    coco_suite = {
        "map50_95": float(val_res.box.map),
        "map50":    float(val_res.box.map50),
        "map75":    float(val_res.box.map75),
        "APs": None, "APm": None, "APl": None,
        "AR_1": None, "AR_10": None, "AR_100": None,
    }

    # Locate predictions JSON written by save_json=True
    pred_json = None
    try:
        if hasattr(args, "save_dir") and args.save_dir:
            candidates = list(Path(str(args.save_dir)).glob("*predictions*.json"))
            if candidates:
                pred_json = candidates[0]
    except Exception:
        pass
    if pred_json is None:
        candidates = list(EXPERIMENTS.glob("**/*predictions*.json"))
        if candidates:
            pred_json = max(candidates, key=lambda p: p.stat().st_mtime)

    if pred_json and pred_json.exists():
        try:
            raw_preds = _json.loads(pred_json.read_text())
            remapped  = []
            for p in raw_preds:
                pid = p["image_id"]
                remapped.append({
                    **p,
                    "image_id": _img_id(str(pid)) if not isinstance(pid, int) else pid
                })
            pred_json_fixed = out_dir / "predictions_fixed.json"
            pred_json_fixed.write_text(_json.dumps(remapped))

            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            coco_gt_obj = COCO(str(coco_gt_json))
            coco_dt_obj = coco_gt_obj.loadRes(str(pred_json_fixed))
            evaluator   = COCOeval(coco_gt_obj, coco_dt_obj, "bbox")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                evaluator.evaluate()
                evaluator.accumulate()
                evaluator.summarize()
            s = evaluator.stats
            coco_suite.update({
                "map50_95": float(s[0]), "map50": float(s[1]), "map75": float(s[2]),
                "APs": float(s[3]), "APm": float(s[4]), "APl": float(s[5]),
                "AR_1": float(s[6]), "AR_10": float(s[7]), "AR_100": float(s[8]),
            })
        except Exception as e:
            print(f"  [WARN] pycocotools: {e}")
    else:
        print(f"  [WARN] predictions.json not found for {run_name}")

    (out_dir / "coco_ap_suite.json").write_text(_json.dumps(coco_suite, indent=2))

    # ── 3. Confusion matrix ───────────────────────────────────────────────────
    conf_out = {}
    try:
        cm     = validator.confusion_matrix
        matrix = cm.matrix.tolist()
        conf_out = {
            "matrix": matrix,
            "class_names": ["polyp"],
            "TP": float(matrix[0][0]),
            "FN": float(matrix[0][1]) if len(matrix[0]) > 1 else None,
            "FP": float(matrix[1][0]) if len(matrix) > 1 else None,
        }
    except Exception as e:
        print(f"  [WARN] confusion matrix: {e}")
        conf_out = {"error": str(e)}

    (out_dir / "confusion_matrix.json").write_text(_json.dumps(conf_out, indent=2))

    def _s(v):
        return f"{v:.4f}" if v is not None else "n/a"

    print(f"  [DONE] {run_name}")
    print(f"    mAP50-95={_s(coco_suite['map50_95'])}  mAP50={_s(coco_suite['map50'])}"
          f"  mAP75={_s(coco_suite['map75'])}")
    print(f"    APs={_s(coco_suite['APs'])}  APm={_s(coco_suite['APm'])}"
          f"  APl={_s(coco_suite['APl'])}")
    return out_dir


print("compute_and_persist_metrics() ready.")
print(f"Metrics dir: {METRICS_DIR}")
""", "colondb-cell-b"))

# ── Cell 36 (Cell C): Run extraction ──────────────────────────────────────────
cells.append(code(
"""# --- Cell C: Run metrics extraction for all completed runs (idempotent)
_all_loss_keys = BASELINE_LOSS_NAMES + [f"aeiou_r{_fmt_r(r)}" for r in AEIOU_RIGIDITIES]
_seed = SEEDS[0]
_total, _done, _skipped, _failed = 0, 0, 0, 0

for _loss_name in _all_loss_keys:
    for _split_name, _yaml_path in SPLIT_CONFIGS.items():
        _run_name = f"colondb_yolo26n_{_loss_name}_{_split_name}_s{_seed}_e{EPOCHS}"
        _weights  = EXPERIMENTS / _run_name / "weights" / "best.pt"
        _total   += 1
        try:
            _result = compute_and_persist_metrics(
                _run_name, _weights, _yaml_path, COCO_GT_JSON
            )
            if _result is None:
                _skipped += 1
            else:
                _done += 1
        except Exception as _e:
            print(f"  [ERROR] {_run_name}: {_e}")
            _failed += 1

print(f"\\nMetrics extraction complete:")
print(f"  computed={_done}  skipped={_skipped}  failed={_failed}  total={_total}")
""", "colondb-cell-c"))

# ── Cell 37 (Cell D): Unified JSON ────────────────────────────────────────────
cells.append(code(
"""# --- Cell D: Build unified metrics_all_losses.json
import json as _json

_all_loss_keys = BASELINE_LOSS_NAMES + [f"aeiou_r{_fmt_r(r)}" for r in AEIOU_RIGIDITIES]
_seed = SEEDS[0]

all_metrics = {}
for _loss_name in _all_loss_keys:
    for _split_name in SPLIT_CONFIGS:
        _run_name = f"colondb_yolo26n_{_loss_name}_{_split_name}_s{_seed}_e{EPOCHS}"
        _mdir = METRICS_DIR / _run_name
        entry = {
            "loss": _loss_name, "split": _split_name, "run": _run_name,
            "rigidity": (float(_loss_name.split("_r")[1].replace("p", "."))
                         if "aeiou" in _loss_name else -1.0),
        }
        for _fname in ["coco_ap_suite.json", "pr_curve.json", "confusion_matrix.json"]:
            _fpath = _mdir / _fname
            if _fpath.exists():
                _data = _json.loads(_fpath.read_text())
                _prefix = {"coco_ap_suite.json": "", "pr_curve.json": "pr_",
                           "confusion_matrix.json": "cm_"}[_fname]
                for _k, _v in _data.items():
                    entry[_prefix + _k] = _v
        all_metrics[_run_name] = entry

_unified_path = EXPERIMENTS / "metrics_all_losses.json"
_unified_path.write_text(_json.dumps(all_metrics, indent=2))
n_runs     = len(all_metrics)
n_complete = sum(1 for v in all_metrics.values() if v.get("map50_95") is not None)
print(f"Unified metrics saved: {_unified_path}")
print(f"  {n_runs} runs total  |  {n_complete} with full COCO AP suite")
""", "colondb-cell-d"))

# ── Cell 38 (Cell E): Cross-loss comparison table ─────────────────────────────
cells.append(code(
"""# --- Cell E: Cross-loss COCO AP suite comparison table (paper-ready)
import pandas as pd

rows = []
for _run_name, m in all_metrics.items():
    if m.get("map50_95") is None:
        continue
    rows.append({
        "loss": m["loss"], "split": m["split"],
        "mAP50-95": m.get("map50_95"), "mAP50": m.get("map50"),
        "mAP75":    m.get("map75"),    "APs":   m.get("APs"),
        "APm":      m.get("APm"),      "APl":   m.get("APl"),
        "AR@100":   m.get("AR_100"),   "TP":    m.get("cm_TP"),
        "FP":       m.get("cm_FP"),    "FN":    m.get("cm_FN"),
    })

df_metrics = pd.DataFrame(rows)

_clean = (df_metrics[df_metrics["split"] == "clean"]
          .drop(columns=["split"]).set_index("loss")
          .sort_values("mAP50-95", ascending=False))

print("=== COCO AP Suite — CVC-ColonDB clean split ===")
print(_clean.round(4).to_string())
_clean.to_csv(ANALYSIS_DIR / "coco_ap_suite_clean.csv")
df_metrics.to_csv(ANALYSIS_DIR / "coco_ap_suite_all_splits.csv", index=False)
print(f"\\nSaved to {ANALYSIS_DIR}")
""", "colondb-cell-e"))

# ── Cell 39: EIoU vs AEIoU summary ────────────────────────────────────────────
cells.append(code(
"""# --- ColonDB EIoU vs AEIoU summary (for cross-dataset analysis)
import pandas as pd

_all_loss_keys = BASELINE_LOSS_NAMES + [f"aeiou_r{_fmt_r(r)}" for r in AEIOU_RIGIDITIES]
eiou_rows = []
for _loss_name in _all_loss_keys:
    for _split_name in SPLIT_CONFIGS:
        _run_name = f"colondb_yolo26n_{_loss_name}_{_split_name}_s{SEEDS[0]}_e{EPOCHS}"
        m = all_metrics.get(_run_name, {})
        if m.get("map50_95") is None:
            continue
        eiou_rows.append({
            "dataset": "colondb",
            "dataset_label": "CVC-ColonDB",
            "dataset_type": "amorphous",
            "n_classes": 1,
            "split": _split_name,
            "loss": _loss_name,
            "map95": m["map50_95"],
            "map50": m.get("map50"),
        })

df_colondb = pd.DataFrame(eiou_rows)

summary_rows = []
for split in ["clean", "low", "high"]:
    sub = df_colondb[df_colondb["split"] == split]
    eiou_row  = sub[sub["loss"] == "eiou"]
    aeiou_sub = sub[sub["loss"].str.startswith("aeiou")]
    if eiou_row.empty or aeiou_sub.empty:
        continue
    eiou_map95 = float(eiou_row["map95"].values[0])
    eiou_map50 = float(eiou_row["map50"].values[0])
    best_aeiou = aeiou_sub.loc[aeiou_sub["map95"].idxmax()]
    best_lam   = float(best_aeiou["loss"].split("_r")[1].replace("p", "."))
    best_map95 = float(best_aeiou["map95"])
    best_map50 = float(best_aeiou["map50"])
    summary_rows.append({
        "dataset": "colondb", "dataset_label": "CVC-ColonDB",
        "dataset_type": "amorphous", "n_classes": 1,
        "split": split,
        "eiou_map95": eiou_map95, "eiou_map50": eiou_map50,
        "best_aeiou_lambda": best_lam,
        "best_aeiou_map95": best_map95, "best_aeiou_map50": best_map50,
        "delta_map95": best_map95 - eiou_map95,
        "delta_map50": best_map50 - eiou_map50,
    })

df_summary = pd.DataFrame(summary_rows)
out_csv = EXPERIMENTS / "colondb_eiou_vs_aeiou_summary.csv"
df_summary.to_csv(out_csv, index=False)
print("=== CVC-ColonDB: EIoU vs Best AEIoU ===")
print(df_summary.to_string(index=False))
print(f"\\nSaved -> {out_csv}")
""", "colondb-eiou-vs-aeiou"))

# ── Cell 40: Final summary ─────────────────────────────────────────────────────
cells.append(code(
"""# --- Final Kaggle output summary
print("=== CVC-ColonDB Run Complete ===")
print(f"Experiment dir : {EXPERIMENTS}")
n_runs_done = len(sorted(EXPERIMENTS.glob("*/results.csv")))
print(f"Completed runs : {n_runs_done} / {len(ALL_LOSS_KEYS)*len(SPLIT_CONFIGS)*len(SEEDS)}")
print()
print("To download outputs:")
print("  Kaggle -> Output tab -> amorphous-yolo/experiments_colondb/")
print("  Key files:")
print("    all_results_combined.csv       — epoch-level training metrics")
print("    metrics_all_losses.json        — COCO AP suite for all runs")
print("    analysis/                      — figures and tables")
print("    colondb_eiou_vs_aeiou_summary.csv — EIoU vs AEIoU cross-split summary")
""", "colondb-final-summary"))

# ── Notebook metadata ──────────────────────────────────────────────────────────
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
            "version": "3.10.12"
        }
    },
    "cells": cells,
}

out_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "05_cvc_colondb_kaggle.ipynb"
)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Written: {out_path}")
print(f"Cells: {len(cells)}")

# Verify JSON is valid
with open(out_path, encoding="utf-8") as f:
    nb_check = json.load(f)
print(f"Verified: {len(nb_check['cells'])} cells, nbformat={nb_check['nbformat']}")
