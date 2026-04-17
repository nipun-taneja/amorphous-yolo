"""
Generate 06_polypgen_kaggle.ipynb — no-touch Kaggle notebook for PolypGen2021.
Key differences from CVC notebooks:
  - Bboxes provided as txt files (PASCAL VOC: polyp x1 y1 x2 y2) — no mask conversion
  - Multi-center (C1–C6) merge before 80/20 split
  - Multi-polyp support (multiple lines per label file)
  - cv2 dims mandatory (C1 EXIF thumbnail trap: 160x120 thumbnail + 1350x1080 main)
  - Skip negatives (empty bbox) and C3 images missing bbox file entirely
Run: python generate_polypgen_notebook.py
"""
import json, os

def _id(tag): return tag[:20]
def code(src, cid): return {"cell_type":"code","execution_count":None,"id":_id(cid),"metadata":{},"outputs":[],"source":src}
def md(src, cid):   return {"cell_type":"markdown","id":_id(cid),"metadata":{},"source":src}

cells = []

# ── Cell 0: Title ──────────────────────────────────────────────────────────────
cells.append(md(
"""# 06 · PolypGen2021: EIoU vs AEIoU — Multi-Center Validation (Kaggle Edition)

**PolypGen2021** is the largest and most diverse polyp detection benchmark:
6 clinical centers, 1347 positive images, pre-provided PASCAL-VOC bounding boxes.

**Before running:**
1. Upload PolypGen2021 as a Kaggle dataset and attach it here.
   Expected input path: `/kaggle/input/<your-slug>/PolypGen2021_MultiCenterData_v3/`
2. Enable GPU: *Settings → Accelerator → GPU T4 x2*
3. Run All — fully automatic.

**Output:** `/kaggle/working/amorphous-yolo/experiments_polypgen/`

| Property | Value |
|---|---|
| Centers | C1–C6 (6 clinical sites) |
| Positive images | ~1347 (negatives skipped) |
| Train / Val | ~1078 / ~269 (80/20, seed 42) |
| Bbox format | Pre-provided PASCAL-VOC txt (no mask conversion) |
| Multi-polyp | Yes (123 images with 2+ boxes) |
| Total runs | 18 losses × 3 splits = **54 runs** |
| Est. runtime | ~5–6 hours on T4 |
""", "pg-title"))

# ── Cell 1: GPU check ──────────────────────────────────────────────────────────
cells.append(code(
"""import os
print('Running on Kaggle. Output dir: /kaggle/working/')
print('GPU:', os.popen('nvidia-smi --query-gpu=name --format=csv,noheader').read().strip())
""", "pg-gpu"))

# ── Cell 2: Install ────────────────────────────────────────────────────────────
cells.append(code(
"""!pip install --upgrade pip -q
!pip install -U "ultralytics==8.4.9" "wandb==0.24.1" "pycocotools" -q
print("Dependencies installed.")
""", "pg-install"))

# ── Cell 3: Git clone ──────────────────────────────────────────────────────────
cells.append(code(
"""import os, sys
REPO_PATH = "/kaggle/working/amorphous-yolo"
if not os.path.exists(f"{REPO_PATH}/.git"):
    os.system(f"git clone https://github.com/nipun-taneja/amorphous-yolo.git {REPO_PATH}")
    print("Cloned.")
else:
    print("Repo already present.")
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
os.chdir(REPO_PATH)
print(f"CWD: {os.getcwd()}")
""", "pg-gitclone"))

# ── Cell 4: Constants ──────────────────────────────────────────────────────────
cells.append(code(
"""import math, time
from pathlib import Path
from datetime import datetime

PROJECT_DIR   = Path("/kaggle/working/amorphous-yolo")
DATASET_ROOT  = PROJECT_DIR / "datasets" / "polypgen"
EXPERIMENTS   = PROJECT_DIR / "experiments_polypgen"
ANALYSIS_DIR  = EXPERIMENTS / "analysis"
MANIFEST_PATH = EXPERIMENTS / "manifest.json"
EXPERIMENTS.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS        = 50       # more epochs: 1347 images converges slower than CVC
IMGSZ         = 640
DEVICE        = 0
MODEL_PT      = "yolo26n.pt"
WANDB_PROJECT = "polypgen-aeiou"
SEEDS         = [42]

CENTERS = ["C1", "C2", "C3", "C4", "C5", "C6"]

BASELINE_LOSS_NAMES = ["iou", "giou", "diou", "ciou", "eiou", "eciou", "siou", "wiou"]
AEIOU_RIGIDITIES    = [round(x * 0.1, 1) for x in range(1, 11)]

def _fmt_r(r): return str(r).replace(".", "p")

ALL_LOSS_KEYS = BASELINE_LOSS_NAMES + [f"aeiou_r{_fmt_r(r)}" for r in AEIOU_RIGIDITIES]
SPLIT_CONFIGS = {}  # populated after YAML creation

PALETTE = {
    "iou":"#888888","giou":"#BC6C25","diou":"#606C38","ciou":"#DDA15E",
    "eiou":"#E63946","eciou":"#9B2226","siou":"#6A0572","wiou":"#FF6B6B",
    "aeiou_r0p1":"#023E8A","aeiou_r0p2":"#0077B6","aeiou_r0p3":"#00B4D8",
    "aeiou_r0p4":"#48CAE4","aeiou_r0p5":"#90E0EF","aeiou_r0p6":"#2A9D8F",
    "aeiou_r0p7":"#52B788","aeiou_r0p8":"#74C69D","aeiou_r0p9":"#95D5B2",
    "aeiou_r1p0":"#6A4C93",
}
LOSS_LABELS = {"iou":"IoU","giou":"GIoU","diou":"DIoU","ciou":"CIoU",
               "eiou":"EIoU","eciou":"ECIoU","siou":"SIoU","wiou":"WIoU"}
for r in AEIOU_RIGIDITIES:
    LOSS_LABELS[f"aeiou_r{_fmt_r(r)}"] = f"AEIoU lam={r}"

n_planned = len(ALL_LOSS_KEYS) * 3 * len(SEEDS)
print(f"Constants loaded. Planned runs: {n_planned}")
print(f"EPOCHS={EPOCHS}  SEEDS={SEEDS}")
""", "pg-constants"))

# ── Cell 5: WandB ─────────────────────────────────────────────────────────────
cells.append(code(
"""import os, wandb
try:
    from kaggle_secrets import UserSecretsClient
    wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"), quiet=True)
    print(f"WandB ready. Project: {WANDB_PROJECT}")
except Exception as e:
    print(f"WandB skipped ({e})")
    os.environ["WANDB_DISABLED"] = "true"
""", "pg-wandb"))

# ── Cell 6: Find PolypGen input ────────────────────────────────────────────────
cells.append(code(
"""# Locate PolypGen2021 under /kaggle/input — handles two zip structures:
#   NESTED: .../data_C1/images_C1/ + data_C1/bbox_C1/   (original PolypGen layout)
#   FLAT:   .../images_C1/ + bbox_C1/                    (flat zip without data_Cx wrapper)
from pathlib import Path

# Print full tree so you can debug if needed
print("Scanning /kaggle/input...")
import os
for root, dirs, files in os.walk("/kaggle/input"):
    depth = root.replace("/kaggle/input","").count(os.sep)
    if depth < 3:  # only show top 3 levels
        print(f"  {'  '*depth}{Path(root).name}/  ({len(files)} files)")

PG_ROOT  = None   # parent that contains data_C1/ (nested layout)
PG_FLAT  = None   # parent that contains images_C1/ directly (flat layout)

for d in sorted(Path("/kaggle/input").rglob("images_C1")):
    if not d.is_dir(): continue
    parent = d.parent
    # Check flat: parent directly has images_C1/ and bbox_C1/
    if (parent / "bbox_C1").is_dir() and len(list((parent/"bbox_C1").glob("*.txt"))) > 0:
        # Nested layout: parent is data_C1, grandparent is PG_ROOT
        if parent.name == "data_C1":
            PG_ROOT = parent.parent
            print(f"Found NESTED layout: PG_ROOT={PG_ROOT}")
        else:
            # Flat layout: parent contains images_C1/ and bbox_C1/ directly
            PG_FLAT = parent
            print(f"Found FLAT layout: PG_FLAT={PG_FLAT}")
        break

assert PG_ROOT is not None or PG_FLAT is not None, (
    "Could not find PolypGen2021 under /kaggle/input.\\n"
    "Attach the dataset. Zip must contain images_C1/ + bbox_C1/ ... images_C6/ + bbox_C6/."
)

# Build per-center dir lookup regardless of layout
def get_center_dirs(c):
    if PG_ROOT is not None:
        return PG_ROOT / f"data_{c}" / f"images_{c}", PG_ROOT / f"data_{c}" / f"bbox_{c}"
    else:
        return PG_FLAT / f"images_{c}", PG_FLAT / f"bbox_{c}"

print()
for c in ["C1","C2","C3","C4","C5","C6"]:
    img_d, bbox_d = get_center_dirs(c)
    n_img  = len(list(img_d.glob("*.jpg")))  if img_d.is_dir()  else 0
    n_bbox = len(list(bbox_d.glob("*.txt"))) if bbox_d.is_dir() else 0
    print(f"  {c}: imgs={n_img}  bbox_files={n_bbox}")
""", "pg-find-input"))

# ── Cell 7: Convert bbox → YOLO labels ────────────────────────────────────────
cells.append(code(
"""# Convert PolypGen PASCAL-VOC bbox txt → YOLO format, merge all centers
# Bbox format: "polyp x1 y1 x2 y2"  (absolute pixels, one or more lines)
# CRITICAL: use cv2 for image dims — C1 JPEGs have 160x120 EXIF thumbnail
#           followed by 1350x1080 main image; PIL reads the thumbnail.
import cv2, random, shutil
from pathlib import Path

MERGED_IMG = DATASET_ROOT / "images_all"
MERGED_LBL = DATASET_ROOT / "labels_all"
MERGED_IMG.mkdir(parents=True, exist_ok=True)
MERGED_LBL.mkdir(parents=True, exist_ok=True)

# Check idempotency
existing = len(list(MERGED_IMG.glob("*.jpg")))
if existing >= 1300:
    print(f"Merged dataset already built ({existing} images) — skipping.")
else:
    skipped_neg = 0     # empty bbox (background images)
    skipped_no_bbox = 0 # C3 images missing bbox file entirely
    skipped_err = 0     # cv2 read failure
    converted   = 0
    multi_polyp = 0

    for c in ["C1","C2","C3","C4","C5","C6"]:
        img_dir, bbox_dir = get_center_dirs(c)

        for img_path in sorted(img_dir.glob("*.jpg")):
            stem   = img_path.stem
            # C1/C4/C5/C6 use {stem}_mask.txt; C2/C3 use {stem}.txt — try both
            bbox_f = bbox_dir / f"{stem}_mask.txt"
            if not bbox_f.exists():
                bbox_f = bbox_dir / f"{stem}.txt"

            # Skip if no bbox file (C3 gap: 64 images)
            if not bbox_f.exists():
                skipped_no_bbox += 1
                continue

            raw = bbox_f.read_text().strip()
            # Skip negatives (empty bbox file)
            if not raw:
                skipped_neg += 1
                continue

            lines = [l for l in raw.splitlines() if l.strip()]

            # Read image dims with cv2 (avoids EXIF thumbnail trap)
            img = cv2.imread(str(img_path))
            if img is None:
                skipped_err += 1
                continue
            H, W = img.shape[:2]

            yolo_lines = []
            valid = True
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5 or parts[0] != "polyp":
                    valid = False; break
                x1, y1, x2, y2 = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                cx = (x1 + x2) / (2 * W)
                cy = (y1 + y2) / (2 * H)
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                # Sanity clip — coords should be in [0,1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                bw = max(0.01, min(1.0, bw))
                bh = max(0.01, min(1.0, bh))
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if not valid or not yolo_lines:
                skipped_err += 1
                continue

            # Unique filename: prefix center to avoid stem collisions across centers
            out_stem = f"{c}_{stem}"
            shutil.copy2(img_path, MERGED_IMG / f"{out_stem}.jpg")
            (MERGED_LBL / f"{out_stem}.txt").write_text("\\n".join(yolo_lines) + "\\n")

            converted += 1
            if len(yolo_lines) > 1:
                multi_polyp += 1

    print(f"Conversion complete:")
    print(f"  Converted  : {converted} images ({multi_polyp} multi-polyp)")
    print(f"  Skipped neg: {skipped_neg}  no_bbox: {skipped_no_bbox}  err: {skipped_err}")

n_imgs = len(list(MERGED_IMG.glob("*.jpg")))
n_lbls = len(list(MERGED_LBL.glob("*.txt")))
print(f"Merged pool: {n_imgs} images  {n_lbls} labels")
assert n_imgs == n_lbls >= 1300, f"Expected >=1300 matched pairs, got {n_imgs}/{n_lbls}"
""", "pg-convert-bbox"))

# ── Cell 8: Train/val split ────────────────────────────────────────────────────
cells.append(code(
"""# 80/20 train/val split (seed=42) — stratified by center for balance
import random, shutil
from pathlib import Path

TRAIN_DIR = DATASET_ROOT / "train"
VALID_DIR = DATASET_ROOT / "valid"

n_train_check = len(list((TRAIN_DIR/"images").glob("*.jpg"))) if (TRAIN_DIR/"images").exists() else 0
if n_train_check >= 1000:
    print(f"Split already exists ({n_train_check} train images) — skipping.")
else:
    for d in [TRAIN_DIR, VALID_DIR]:
        (d/"images").mkdir(parents=True, exist_ok=True)
        (d/"labels").mkdir(parents=True, exist_ok=True)

    all_imgs = sorted(MERGED_IMG.glob("*.jpg"))

    # Stratified 80/20 by center prefix
    from collections import defaultdict
    by_center = defaultdict(list)
    for p in all_imgs:
        center = p.stem.split("_")[0]  # e.g. "C1"
        by_center[center].append(p)

    train_imgs, val_imgs = [], []
    random.seed(42)
    for center, imgs in sorted(by_center.items()):
        random.shuffle(imgs)
        k = int(0.8 * len(imgs))
        train_imgs.extend(imgs[:k])
        val_imgs.extend(imgs[k:])
        print(f"  {center}: {k} train  {len(imgs)-k} val")

    for split_name, img_list in [("train", train_imgs), ("valid", val_imgs)]:
        out_img = DATASET_ROOT / split_name / "images"
        out_lbl = DATASET_ROOT / split_name / "labels"
        for img_path in img_list:
            lbl_path = MERGED_LBL / f"{img_path.stem}.txt"
            shutil.copy2(img_path, out_img / img_path.name)
            if lbl_path.exists():
                shutil.copy2(lbl_path, out_lbl / f"{img_path.stem}.txt")

    n_tr = len(list((TRAIN_DIR/"images").glob("*.jpg")))
    n_vl = len(list((VALID_DIR/"images").glob("*.jpg")))
    print(f"Split done: {n_tr} train  {n_vl} val")
    assert n_tr >= 1000, f"Expected >=1000 train, got {n_tr}"

print("Train/val split ready.")
""", "pg-split"))

# ── Cell 9: Noise splits ───────────────────────────────────────────────────────
cells.append(code(
"""import numpy as np, shutil
from pathlib import Path

SIGMA_LOW  = 0.02
SIGMA_HIGH = 0.08

def _jitter_label_file(src_lbl, dst_lbl, sigma, rng):
    lines = Path(src_lbl).read_text().strip().split("\\n")
    out = []
    for line in lines:
        if not line.strip(): continue
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
    src_img = root / "valid" / "images"
    src_lbl = root / "valid" / "labels"
    rng = np.random.default_rng(seed)
    for split_name, sigma in [("valid_low", SIGMA_LOW), ("valid_high", SIGMA_HIGH)]:
        dst_img = root / split_name / "images"
        dst_lbl = root / split_name / "labels"
        n_ex = len(list(dst_img.glob("*.jpg"))) if dst_img.exists() else 0
        if n_ex >= 200:
            print(f"  {split_name}: exists ({n_ex}) — skip"); continue
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        for img_path in sorted(src_img.glob("*.jpg")):
            stem = img_path.stem
            dst = dst_img / img_path.name
            if not dst.exists():
                try: dst.symlink_to(img_path.resolve())
                except (OSError, NotImplementedError): shutil.copy2(img_path, dst)
            sl = src_lbl / f"{stem}.txt"
            if sl.exists():
                _jitter_label_file(sl, dst_lbl / f"{stem}.txt", sigma, rng)
        print(f"  {split_name}: {len(list(dst_img.glob('*.jpg')))} images (sigma={sigma})")

build_noise_splits(DATASET_ROOT)
print("Noise splits ready.")
""", "pg-noise"))

# ── Cell 10: YAML files ────────────────────────────────────────────────────────
cells.append(code(
"""from pathlib import Path

DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

yaml_defs = {
    "polypgen.yaml":      "valid/images",
    "polypgen_low.yaml":  "valid_low/images",
    "polypgen_high.yaml": "valid_high/images",
}
for fname, val_split in yaml_defs.items():
    content = (
        f"path: {DATASET_ROOT}\\n"
        f"train: train/images\\n"
        f"val: {val_split}\\n"
        f"nc: 1\\n"
        f"names: ['polyp']\\n"
    )
    (DATA_DIR / fname).write_text(content)
    print(f"Created: {DATA_DIR/fname}")

SPLIT_CONFIGS = {
    "clean": DATA_DIR / "polypgen.yaml",
    "low":   DATA_DIR / "polypgen_low.yaml",
    "high":  DATA_DIR / "polypgen_high.yaml",
}
""", "pg-yaml"))

# ── Cell 11: Verify ───────────────────────────────────────────────────────────
cells.append(code(
"""import yaml
from pathlib import Path

print(f"{'Split':<8} {'Val images':>12}")
for split_name, cfg_path in SPLIT_CONFIGS.items():
    cfg    = yaml.safe_load(cfg_path.read_text())
    val_d  = DATASET_ROOT / cfg["val"]
    n      = len(list(val_d.glob("*.jpg")))
    status = "OK" if n >= 200 else "CHECK"
    print(f"{split_name:<8} {n:>12}  {status}")

n_tr = len(list((TRAIN_DIR/"images").glob("*.jpg")))
print(f"\\nTrain: {n_tr}")
assert n_tr >= 1000
print("All splits verified.")
""", "pg-verify"))

# ── Cell 12: Monkey-patch ─────────────────────────────────────────────────────
cells.append(code(
"""import torch, torch.nn.functional as F
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
            target_ltrb = loss_mod.bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask],
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = loss_mod.bbox2dist(anchor_points, target_bboxes) * stride
            target_ltrb[..., 0::2] /= imgsz[1]; target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist_s = pred_dist * stride
            pred_dist_s[..., 0::2] /= imgsz[1]; pred_dist_s[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist_s[fg_mask], target_ltrb[fg_mask],
                          reduction="none").mean(-1, keepdim=True) * weight
            ).sum() / target_scores_sum
        return loss_iou, loss_dfl
    return bbox_loss_forward

def patch_loss(loss_fn_instance):
    loss_mod.BboxLoss.forward = _make_bbox_forward(loss_fn_instance)
    rig = getattr(loss_fn_instance, "rigidity", None)
    print(f"  [PATCH] -> {type(loss_fn_instance).__name__}" + (f"(lam={rig})" if rig else ""))

def restore_loss():
    loss_mod.BboxLoss.forward = _ORIGINAL_BBOX_FORWARD

print("Patch ready.")
""", "pg-patch"))

# ── Cell 13: Patch verify ─────────────────────────────────────────────────────
cells.append(code(
"""from src.losses import EIoULoss
patch_loss(EIoULoss(reduction="none"))
restore_loss()
assert loss_mod.BboxLoss.forward is _ORIGINAL_BBOX_FORWARD
print("Patch round-trip OK.")
""", "pg-patch-verify"))

# ── Cell 14: Loss registry ────────────────────────────────────────────────────
cells.append(code(
"""from src.losses import (IoULoss, GIoULoss, DIoULoss, CIoULoss,
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
print(f"Baselines: {list(BASELINE_LOSS_REGISTRY)}")
print(f"AEIoU: lambda={AEIOU_RIGIDITIES}")
print(f"Total planned: {len(ALL_LOSS_REGISTRY)*len(SPLIT_CONFIGS)*len(SEEDS)} runs")
""", "pg-loss-registry"))

# ── Cell 15: Training function ────────────────────────────────────────────────
cells.append(code(
"""import json
from pathlib import Path as _Path
from datetime import datetime
from ultralytics import YOLO

def _load_manifest():
    return json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}

def write_manifest_entry(run_name, meta):
    m = _load_manifest(); m[run_name] = meta
    MANIFEST_PATH.write_text(json.dumps(m, indent=2))

def run_training(loss_name, loss_fn, split_name, yaml_path,
                 seed=42, epochs=None, imgsz=None, device=None):
    epochs = epochs or EPOCHS; imgsz = imgsz or IMGSZ; device = device if device is not None else DEVICE
    run_name = f"polypgen_yolo26n_{loss_name}_{split_name}_s{seed}_e{epochs}"
    run_dir  = EXPERIMENTS / run_name

    csv_path = run_dir / "results.csv"
    if csv_path.exists():
        import pandas as _pd
        if len(_pd.read_csv(csv_path)) >= epochs:
            print(f"[SKIP] {run_name}"); return run_dir

    local_last = run_dir / "weights" / "last.pt"
    resuming   = local_last.exists() and not csv_path.exists()

    print(f"\\n{'='*68}")
    print(f"[{'RESUME' if resuming else 'START '}] {run_name}")
    if not resuming:
        print(f"  loss={loss_name}  split={split_name}  seed={seed}  epochs={epochs}")
    print(f"{'='*68}")

    meta = {"loss": loss_name, "split": split_name, "seed": seed, "epochs": epochs,
            "rigidity": float(getattr(loss_fn, "rigidity", -1) or -1),
            "run_dir": str(run_dir), "timestamp": datetime.now().isoformat(),
            "status": "running", "resumed": resuming}
    write_manifest_entry(run_name, meta)
    t0 = time.time()

    try:
        import os as _os
        _os.environ.update({"WANDB_PROJECT": WANDB_PROJECT, "WANDB_NAME": run_name,
                            "WANDB_TAGS": f"{loss_name},{split_name}"})
        patch_loss(loss_fn)

        if resuming:
            results = YOLO(str(local_last)).train(resume=True)
        else:
            results = YOLO(MODEL_PT).train(
                data=str(yaml_path), epochs=epochs, imgsz=imgsz,
                project=str(EXPERIMENTS), name=run_name,
                device=device, seed=seed, exist_ok=True,
            )
        try:
            (run_dir/"run_meta.json").write_text(json.dumps(results.results_dict, indent=2))
        except Exception as e:
            print(f"  [WARN] run_meta: {e}")

        meta["status"] = "complete"; meta["elapsed_sec"] = round(time.time()-t0, 1)
        try:
            import wandb as _w
            if _w.run: _w.finish()
        except Exception: pass

    except Exception as e:
        print(f"  [ERROR] {run_name}: {e}")
        meta["status"] = "failed"; meta["error"] = str(e); raise
    finally:
        restore_loss(); write_manifest_entry(run_name, meta)

    print(f"[DONE] {run_name}  ({meta.get('elapsed_sec',0):.0f}s)")
    return run_dir

print("run_training() ready.")
""", "pg-train-fn"))

# ── Cell 16: Baseline training ────────────────────────────────────────────────
cells.append(md("## Section 7 · Baseline Training\n\n8 losses × 3 splits × 1 seed = **24 runs** (~2.5 hrs)\n", "pg-sec7"))
cells.append(code(
"""print(f"Baselines: {len(BASELINE_LOSS_REGISTRY)} losses x {len(SPLIT_CONFIGS)} splits x {len(SEEDS)} seed(s)")
for loss_name, loss_fn in BASELINE_LOSS_REGISTRY.items():
    for split_name, cfg_path in SPLIT_CONFIGS.items():
        for seed in SEEDS:
            run_training(loss_name=loss_name, loss_fn=loss_fn,
                         split_name=split_name, yaml_path=cfg_path, seed=seed)
restore_loss()
print("Baselines complete.")
""", "pg-baseline-train"))

# ── Cell 17: AEIoU grid ───────────────────────────────────────────────────────
cells.append(md("## Section 8 · AEIoU Rigidity Grid\n\n10 λ × 3 splits × 1 seed = **30 runs** (~3 hrs)\n", "pg-sec8"))
cells.append(code(
"""total = len(AEIOU_RIGIDITIES) * len(SPLIT_CONFIGS) * len(SEEDS)
done  = 0
print(f"AEIoU grid: {total} runs")
for r in AEIOU_RIGIDITIES:
    loss_name = f"aeiou_r{_fmt_r(r)}"
    loss_fn   = AEIOU_LOSS_REGISTRY[loss_name]
    for split_name, cfg_path in SPLIT_CONFIGS.items():
        for seed in SEEDS:
            done += 1
            print(f"[{done}/{total}] lam={r}  split={split_name}")
            run_training(loss_name=loss_name, loss_fn=loss_fn,
                         split_name=split_name, yaml_path=cfg_path, seed=seed)
restore_loss()
print("AEIoU grid complete.")
""", "pg-aeiou-train"))

# ── Cell 18: Load results ─────────────────────────────────────────────────────
cells.append(md("## Section 9 · Results\n", "pg-sec9"))
cells.append(code(
"""import pandas as pd

CACHE_CSV = EXPERIMENTS / "all_results_combined.csv"

def load_all_results(force_rebuild=False):
    if CACHE_CSV.exists() and not force_rebuild:
        return pd.read_csv(CACHE_CSV)
    dfs = []
    for loss_name in ALL_LOSS_KEYS:
        for split_name in SPLIT_CONFIGS:
            for seed in SEEDS:
                run_name = f"polypgen_yolo26n_{loss_name}_{split_name}_s{seed}_e{EPOCHS}"
                csv_path = EXPERIMENTS / run_name / "results.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    df.columns = df.columns.str.strip()
                    df["run_name"] = run_name; df["loss"] = loss_name
                    df["split"] = split_name; df["seed"] = seed
                    df["rigidity"] = (float(loss_name.split("_r")[1].replace("p","."))
                                      if "aeiou" in loss_name else -1.0)
                    dfs.append(df)
                else:
                    print(f"  [MISSING] {run_name}")
    if not dfs: raise RuntimeError("No results found.")
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(CACHE_CSV, index=False)
    return df_all

df_all = load_all_results(force_rebuild=True)
print(f"Shape: {df_all.shape}  |  Runs: {df_all['run_name'].nunique()}/{len(ALL_LOSS_KEYS)*len(SPLIT_CONFIGS)*len(SEEDS)}")
""", "pg-load-results"))

# ── Cell 19: Summary table ────────────────────────────────────────────────────
cells.append(code(
"""import pandas as pd, numpy as np

MAP95 = "metrics/mAP50-95(B)"
MAP50 = "metrics/mAP50(B)"

df_final = df_all.sort_values("epoch").groupby("run_name").last().reset_index()

agg_rows = []
for loss_name in ALL_LOSS_KEYS:
    for split in ["clean","low","high"]:
        sub = df_final[(df_final["loss"]==loss_name) & (df_final["split"]==split)]
        if sub.empty: continue
        row = {"loss": loss_name, "split": split, "label": LOSS_LABELS.get(loss_name, loss_name)}
        for col, tag in [(MAP95,"map95"),(MAP50,"map50")]:
            if col in sub.columns:
                row[f"{tag}_mean"] = sub[col].mean()
                row[f"{tag}_std"]  = sub[col].std() if len(sub)>1 else 0.0
        agg_rows.append(row)

df_agg = pd.DataFrame(agg_rows)

pivot95 = df_agg.pivot_table(index="loss", columns="split", values="map95_mean")
pivot95 = pivot95[["clean","low","high"]]
pivot95["robust_ratio"] = pivot95["high"] / pivot95["clean"].replace(0, float("nan"))
pivot95 = pivot95.sort_values("clean", ascending=False)

pivot50 = df_agg.pivot_table(index="loss", columns="split", values="map50_mean")
pivot50 = pivot50[["clean","low","high"]]

pivot95.to_csv(ANALYSIS_DIR / "summary_map95.csv")
pivot50.to_csv(ANALYSIS_DIR / "summary_map50.csv")

print("=== mAP50-95 by Loss and Split ===")
print(pivot95.round(4).to_string())
print("\\n=== mAP50 by Loss and Split ===")
print(pivot50.round(4).to_string())
""", "pg-summary"))

# ── Cell 20: Figures ──────────────────────────────────────────────────────────
cells.append(md("## Section 10 · Figures\n", "pg-sec10"))

cells.append(code(
"""import matplotlib.pyplot as plt, numpy as np

# Fig 1: Bar chart
x_labels = BASELINE_LOSS_NAMES + [f"aeiou_r{_fmt_r(r)}" for r in AEIOU_RIGIDITIES]
disp_lbl  = [LOSS_LABELS.get(l,l) for l in x_labels]
sc = {"clean":"#4CAF50","low":"#FFC107","high":"#F44336"}
x, w = np.arange(len(x_labels)), 0.25
fig, ax = plt.subplots(figsize=(18,6))
for i, split in enumerate(["clean","low","high"]):
    vals = [pivot95.loc[l,split] if l in pivot95.index else 0 for l in x_labels]
    ax.bar(x+i*w, vals, w, color=sc[split], alpha=0.85, label=split)
ax.axvline(x=len(BASELINE_LOSS_NAMES)-0.5, color="black", linestyle=":", lw=1.5, alpha=0.4)
ax.set_xticks(x+w); ax.set_xticklabels(disp_lbl, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("mAP@[.5:.95]"); ax.legend(title="Split")
ax.set_title("PolypGen2021 — Final mAP50-95: All Losses × 3 Splits", fontweight="bold")
ax.grid(axis="y", alpha=0.3); plt.tight_layout()
plt.savefig(ANALYSIS_DIR/"01_final_map95_comparison.png", dpi=150, bbox_inches="tight")
plt.show(); print("Saved 01_final_map95_comparison.png")
""", "pg-fig1"))

cells.append(code(
"""# Fig 2: Lambda sweep
fig, axes = plt.subplots(1, 2, figsize=(14,5))
fig.suptitle("PolypGen2021 — AEIoU Rigidity Sweep vs Baselines", fontweight="bold")
for ax, mc, yl in [(axes[0],"map95_mean","mAP50-95"),(axes[1],"map50_mean","mAP50")]:
    for split, ls, mk in [("clean","-","o"),("low","--","s"),("high",":","^")]:
        lams, maps = [], []
        for r in AEIOU_RIGIDITIES:
            row = df_agg[(df_agg["loss"]==f"aeiou_r{_fmt_r(r)}")&(df_agg["split"]==split)]
            if not row.empty: lams.append(r); maps.append(row[mc].values[0])
        if lams: ax.plot(lams, maps, ls+mk, color="#0077B6", lw=2, markersize=5, label=f"AEIoU ({split})")
    for bn in BASELINE_LOSS_NAMES:
        row = df_agg[(df_agg["loss"]==bn)&(df_agg["split"]=="clean")]
        if not row.empty:
            val = row[mc].values[0]
            ax.axhline(val, color=PALETTE.get(bn,"#888"), ls="--", lw=1.0, alpha=0.7,
                       label=f"{LOSS_LABELS.get(bn,bn)} ({val:.3f})")
    ax.set_xlabel("lambda"); ax.set_ylabel(yl)
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(ANALYSIS_DIR/"02_lambda_vs_map.png", dpi=150, bbox_inches="tight")
plt.show(); print("Saved 02_lambda_vs_map.png")
""", "pg-fig2"))

cells.append(code(
"""# Fig 3: Learning curves (clean split)
MAP95_COL = "metrics/mAP50-95(B)"; LOSS_COL = "train/box_loss"
vis = BASELINE_LOSS_NAMES + ["aeiou_r0p1","aeiou_r0p3","aeiou_r0p5","aeiou_r1p0"]
fig, axes = plt.subplots(2,1,figsize=(14,8),sharex=True)
fig.suptitle("PolypGen2021 — Learning Curves (clean split)", fontweight="bold")
for ln in vis:
    color=PALETTE.get(ln,"#888"); label=LOSS_LABELS.get(ln,ln)
    sub = df_all[(df_all["loss"]==ln)&(df_all["split"]=="clean")]
    if sub.empty: continue
    avg = sub.groupby("epoch").mean(numeric_only=True).reset_index()
    lw = 2.2 if ln in ["eiou","eciou"] or "aeiou" in ln else 1.2
    ls = "-" if ln in BASELINE_LOSS_NAMES else "--"
    if LOSS_COL in avg.columns: axes[0].plot(avg["epoch"],avg[LOSS_COL],color=color,lw=lw,ls=ls,label=label)
    if MAP95_COL in avg.columns: axes[1].plot(avg["epoch"],avg[MAP95_COL],color=color,lw=lw,ls=ls,label=label)
for ax in axes: ax.legend(fontsize=7,ncol=3); ax.grid(alpha=0.3)
axes[0].set_ylabel("Train box loss"); axes[1].set_ylabel("Val mAP50-95"); axes[1].set_xlabel("Epoch")
plt.tight_layout()
plt.savefig(ANALYSIS_DIR/"03_learning_curves.png", dpi=150, bbox_inches="tight")
plt.show(); print("Saved 03_learning_curves.png")
""", "pg-fig3"))

cells.append(code(
"""# Fig 4: Noise robustness gap
all_losses = BASELINE_LOSS_NAMES + [f"aeiou_r{_fmt_r(r)}" for r in AEIOU_RIGIDITIES]
gaps = [float(pivot95.loc[l,"clean"])-float(pivot95.loc[l,"high"]) if l in pivot95.index else 0 for l in all_losses]
disp = [LOSS_LABELS.get(l,l) for l in all_losses]
cols = [PALETTE.get(l,"#888") for l in all_losses]
fig, ax = plt.subplots(figsize=(16,5))
ax.bar(disp, gaps, color=cols, edgecolor="white", lw=0.5)
if "eiou" in all_losses:
    eg = gaps[all_losses.index("eiou")]
    ax.axhline(eg, color="#E63946", ls="--", lw=1.5, label=f"EIoU gap={eg:.4f}")
ax.set_ylabel("mAP gap (clean-high)  lower=more robust")
ax.set_title("PolypGen2021 — Noise Robustness Gap", fontweight="bold")
ax.tick_params(axis="x", rotation=45); ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(ANALYSIS_DIR/"04_noise_robustness_gap.png", dpi=150, bbox_inches="tight")
plt.show()

import pandas as pd
rank = [{"loss":l,"label":LOSS_LABELS.get(l,l),
         "clean":float(pivot95.loc[l,"clean"]),"high":float(pivot95.loc[l,"high"]),
         "gap":float(pivot95.loc[l,"clean"])-float(pivot95.loc[l,"high"]),
         "ratio":float(pivot95.loc[l,"high"])/float(pivot95.loc[l,"clean"])}
        for l in all_losses if l in pivot95.index]
rank_df = pd.DataFrame(rank).sort_values("ratio", ascending=False)
rank_df.to_csv(ANALYSIS_DIR/"robustness_ranking.csv", index=False)
print("\\n=== Robustness Ranking ===")
print(rank_df.round(4).to_string(index=False))
""", "pg-fig4"))

# ── Cell 21: Section 14 metrics extraction ─────────────────────────────────────
cells.append(md(
"""## Section 14 · COCO Metrics Extraction

Cell A: build GT JSON | Cell B: DetectionValidator + pycocotools | Cell C: run all | Cell D/E: collate
""", "pg-sec14"))

# Cell A
cells.append(code(
"""import cv2, json as _json, hashlib
from pathlib import Path

def _img_id(stem: str) -> int:
    if stem.isdigit(): return int(stem)
    return int(hashlib.md5(stem.encode()).hexdigest(), 16) % (2**31)

def build_coco_gt_json(val_img_dir, val_lbl_dir, out_path, category_name="polyp"):
    if Path(out_path).exists():
        print(f"GT JSON exists: {out_path}"); return
    images, annotations, ann_id = [], [], 1
    for img_path in sorted(Path(val_img_dir).glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None: continue
        H, W = img.shape[:2]
        stem = img_path.stem; iid = _img_id(stem)
        images.append({"id":iid,"file_name":img_path.name,"width":W,"height":H})
        lbl = Path(val_lbl_dir) / f"{stem}.txt"
        if not lbl.exists(): continue
        for line in lbl.read_text().strip().splitlines():
            if not line.strip(): continue
            _, cx, cy, bw, bh = map(float, line.split())
            x1=(cx-bw/2)*W; y1=(cy-bh/2)*H; wa=bw*W; ha=bh*H
            annotations.append({"id":ann_id,"image_id":iid,"category_id":1,
                                 "bbox":[x1,y1,wa,ha],"area":wa*ha,"iscrowd":0})
            ann_id += 1
    coco = {"images":images,"annotations":annotations,
            "categories":[{"id":1,"name":category_name}]}
    Path(out_path).write_text(_json.dumps(coco))
    print(f"GT JSON: {out_path}  ({len(images)} imgs, {len(annotations)} anns)")

COCO_GT_JSON = DATASET_ROOT / "valid" / "coco_gt.json"
build_coco_gt_json(DATASET_ROOT/"valid"/"images", DATASET_ROOT/"valid"/"labels", COCO_GT_JSON)
""", "pg-cell-a"))

# Cell B
cells.append(code(
"""import hashlib, io, contextlib, json as _json
import numpy as np
from pathlib import Path
from ultralytics.cfg import get_cfg, DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionValidator

METRICS_DIR = EXPERIMENTS / "metrics"
METRICS_DIR.mkdir(exist_ok=True)

def compute_and_persist_metrics(run_name, weights_path, yaml_path, coco_gt_json, force=False):
    out_dir = METRICS_DIR / run_name
    done    = out_dir / "coco_ap_suite.json"
    if done.exists() and not force: print(f"  [SKIP] {run_name}"); return None
    if not Path(weights_path).exists(): print(f"  [MISS] {weights_path}"); return None
    out_dir.mkdir(exist_ok=True)

    args = get_cfg(DEFAULT_CFG, dict(model=str(weights_path), data=str(yaml_path),
                   conf=0.001, iou=0.6, verbose=False, save_json=True, imgsz=640, split="val"))
    validator = DetectionValidator(args=args); validator()
    val_res = validator.metrics

    # PR curve
    try:
        prec = np.atleast_2d(np.array(val_res.box.p))[0].tolist()
        rec  = np.atleast_2d(np.array(val_res.box.r))[0].tolist()
        f1   = np.atleast_2d(np.array(val_res.box.f1))[0].tolist()
    except Exception: prec,rec,f1=[],[],[]
    try: ap_per_iou = np.atleast_2d(val_res.box.ap)[0].tolist()
    except Exception: ap_per_iou = []
    (out_dir/"pr_curve.json").write_text(_json.dumps({
        "precision":prec,"recall":rec,"f1":f1,
        "ap50":float(val_res.box.map50),"ap75":float(val_res.box.map75),
        "map50_95":float(val_res.box.map),
        "ap_per_iou_threshold":ap_per_iou,
        "iou_thresholds":np.round(np.arange(0.5,1.0,0.05),2).tolist(),
    }, indent=2))

    # COCO AP suite via pycocotools
    coco_suite = {"map50_95":float(val_res.box.map),"map50":float(val_res.box.map50),
                  "map75":float(val_res.box.map75),
                  "APs":None,"APm":None,"APl":None,"AR_1":None,"AR_10":None,"AR_100":None}
    pred_json = None
    try:
        if hasattr(args,"save_dir") and args.save_dir:
            cands = list(Path(str(args.save_dir)).glob("*predictions*.json"))
            if cands: pred_json = cands[0]
    except Exception: pass
    if pred_json is None:
        cands = list(EXPERIMENTS.glob("**/*predictions*.json"))
        if cands: pred_json = max(cands, key=lambda p: p.stat().st_mtime)

    if pred_json and pred_json.exists():
        try:
            raw = _json.loads(pred_json.read_text())
            remapped = [{**p,"image_id":(_img_id(str(p["image_id"])) if not isinstance(p["image_id"],int) else p["image_id"])} for p in raw]
            fixed = out_dir/"predictions_fixed.json"
            fixed.write_text(_json.dumps(remapped))
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            gt_obj = COCO(str(coco_gt_json)); dt_obj = gt_obj.loadRes(str(fixed))
            ev = COCOeval(gt_obj, dt_obj, "bbox")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf): ev.evaluate(); ev.accumulate(); ev.summarize()
            s = ev.stats
            coco_suite.update({"map50_95":float(s[0]),"map50":float(s[1]),"map75":float(s[2]),
                               "APs":float(s[3]),"APm":float(s[4]),"APl":float(s[5]),
                               "AR_1":float(s[6]),"AR_10":float(s[7]),"AR_100":float(s[8])})
        except Exception as e: print(f"  [WARN] pycocotools: {e}")
    else: print(f"  [WARN] predictions.json missing for {run_name}")

    (out_dir/"coco_ap_suite.json").write_text(_json.dumps(coco_suite, indent=2))

    # Confusion matrix
    try:
        cm = validator.confusion_matrix; mat = cm.matrix.tolist()
        cf = {"matrix":mat,"class_names":["polyp"],
              "TP":float(mat[0][0]),
              "FN":float(mat[0][1]) if len(mat[0])>1 else None,
              "FP":float(mat[1][0]) if len(mat)>1 else None}
    except Exception as e: cf = {"error":str(e)}
    (out_dir/"confusion_matrix.json").write_text(_json.dumps(cf, indent=2))

    s = lambda v: f"{v:.4f}" if v is not None else "n/a"
    print(f"  [DONE] {run_name}  mAP95={s(coco_suite['map50_95'])}  mAP50={s(coco_suite['map50'])}  APs={s(coco_suite['APs'])}")
    return out_dir

print("compute_and_persist_metrics() ready.")
""", "pg-cell-b"))

# Cell C
cells.append(code(
"""# Cell C: run metrics for all completed runs
_total, _done, _skip, _fail = 0, 0, 0, 0
for _ln in ALL_LOSS_KEYS:
    for _sn, _yp in SPLIT_CONFIGS.items():
        _rn = f"polypgen_yolo26n_{_ln}_{_sn}_s{SEEDS[0]}_e{EPOCHS}"
        _w  = EXPERIMENTS / _rn / "weights" / "best.pt"
        _total += 1
        try:
            _r = compute_and_persist_metrics(_rn, _w, _yp, COCO_GT_JSON)
            if _r is None: _skip += 1
            else: _done += 1
        except Exception as _e:
            print(f"  [ERROR] {_rn}: {_e}"); _fail += 1
print(f"\\nDone: computed={_done}  skipped={_skip}  failed={_fail}  total={_total}")
""", "pg-cell-c"))

# Cell D
cells.append(code(
"""# Cell D: unified metrics JSON
import json as _json
all_metrics = {}
for _ln in ALL_LOSS_KEYS:
    for _sn in SPLIT_CONFIGS:
        _rn = f"polypgen_yolo26n_{_ln}_{_sn}_s{SEEDS[0]}_e{EPOCHS}"
        _md = METRICS_DIR / _rn
        entry = {"loss":_ln,"split":_sn,"run":_rn,
                 "rigidity":(float(_ln.split("_r")[1].replace("p",".")) if "aeiou" in _ln else -1.0)}
        for _fn in ["coco_ap_suite.json","pr_curve.json","confusion_matrix.json"]:
            _fp = _md / _fn
            if _fp.exists():
                _pfx = {"coco_ap_suite.json":"","pr_curve.json":"pr_","confusion_matrix.json":"cm_"}[_fn]
                for _k,_v in _json.loads(_fp.read_text()).items():
                    entry[_pfx+_k] = _v
        all_metrics[_rn] = entry
up = EXPERIMENTS/"metrics_all_losses.json"
up.write_text(_json.dumps(all_metrics, indent=2))
nc = sum(1 for v in all_metrics.values() if v.get("map50_95") is not None)
print(f"Unified: {len(all_metrics)} runs  {nc} with full COCO suite -> {up}")
""", "pg-cell-d"))

# Cell E
cells.append(code(
"""# Cell E: paper-ready table + cross-dataset summary
import pandas as pd

rows = [{"loss":m["loss"],"split":m["split"],
         "mAP50-95":m.get("map50_95"),"mAP50":m.get("map50"),
         "mAP75":m.get("map75"),"APs":m.get("APs"),"APm":m.get("APm"),"APl":m.get("APl"),
         "AR@100":m.get("AR_100"),"TP":m.get("cm_TP"),"FP":m.get("cm_FP"),"FN":m.get("cm_FN")}
        for m in all_metrics.values() if m.get("map50_95") is not None]
df_m = pd.DataFrame(rows)
clean = df_m[df_m["split"]=="clean"].drop(columns=["split"]).set_index("loss").sort_values("mAP50-95",ascending=False)
print("=== COCO AP Suite — PolypGen clean split ===")
print(clean.round(4).to_string())
clean.to_csv(ANALYSIS_DIR/"coco_ap_suite_clean.csv")
df_m.to_csv(ANALYSIS_DIR/"coco_ap_suite_all_splits.csv", index=False)

# EIoU vs AEIoU summary (for cross-dataset CSV)
summary_rows = []
for split in ["clean","low","high"]:
    sub = df_m[df_m["split"]==split]
    er  = sub[sub["loss"]=="eiou"]
    ar  = sub[sub["loss"].str.startswith("aeiou")]
    if er.empty or ar.empty: continue
    best = ar.loc[ar["mAP50-95"].idxmax()]
    lam  = float(best["loss"].split("_r")[1].replace("p","."))
    summary_rows.append({
        "dataset":"polypgen","dataset_label":"PolypGen2021",
        "dataset_type":"amorphous","n_classes":1,"split":split,
        "eiou_map95":float(er["mAP50-95"].values[0]),"eiou_map50":float(er["mAP50"].values[0]),
        "best_aeiou_lambda":lam,
        "best_aeiou_map95":float(best["mAP50-95"]),"best_aeiou_map50":float(best["mAP50"]),
        "delta_map95":float(best["mAP50-95"])-float(er["mAP50-95"].values[0]),
        "delta_map50":float(best["mAP50"])-float(er["mAP50"].values[0]),
    })
df_sum = pd.DataFrame(summary_rows)
out = EXPERIMENTS/"polypgen_eiou_vs_aeiou_summary.csv"
df_sum.to_csv(out, index=False)
print("\\n=== PolypGen: EIoU vs Best AEIoU ===")
print(df_sum.to_string(index=False))
print(f"Saved -> {out}")
""", "pg-cell-e"))

# ── Cell final ────────────────────────────────────────────────────────────────
cells.append(code(
"""print("=== PolypGen2021 Run Complete ===")
n_done = len(sorted(EXPERIMENTS.glob("*/results.csv")))
print(f"Runs complete: {n_done} / {len(ALL_LOSS_KEYS)*len(SPLIT_CONFIGS)*len(SEEDS)}")
print()
print("Download: Kaggle -> Output tab -> amorphous-yolo/experiments_polypgen/")
print("Key files:")
print("  all_results_combined.csv          epoch-level metrics")
print("  metrics_all_losses.json           COCO AP suite all runs")
print("  analysis/                         figures + tables")
print("  polypgen_eiou_vs_aeiou_summary.csv  cross-dataset row")
""", "pg-final"))

# ── Write notebook ─────────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10.12"}
    },
    "cells": cells,
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "06_polypgen_kaggle.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

# Verify
with open(out_path, encoding="utf-8") as f:
    nb = json.load(f)
print(f"Written: {out_path}")
print(f"Cells: {len(nb['cells'])}  nbformat={nb['nbformat']}")
