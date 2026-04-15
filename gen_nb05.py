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
# 05 · COCO 2017: Amorphous vs Rigid Object Subsets

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nipun-taneja/amorphous-yolo/blob/main/notebooks/05_coco_amorphous_rigid.ipynb)

**Purpose:** Test AEIoU on COCO 2017 filtered into two complementary subsets:
- **Amorphous** (15 classes): animals + food items with deformable, irregular boundaries
- **Rigid** (10 classes): vehicles + electronics with precise, well-defined boundaries

**Key experimental question:** Does AEIoU help on amorphous objects but NOT hurt on rigid ones?
Expected: large AEIoU delta on amorphous subset, near-zero delta on rigid control.

**Runtime:** ~54 h on A100 across multiple sessions (resume logic handles disconnections).
Run simultaneously with notebook 04 to halve total wall-clock time.
"""), "cell-00-title"))

# Cell 1: Env setup markdown
cells.append(md_cell(src(r"""
## Section 1 · Environment Setup

**Requirements**
- Google Colab A100 (recommended) — this notebook runs ~54h total
- `fiftyone` library for programmatic COCO subset download (no 25 GB full download)
- Colab Secrets: `WANDB_API_KEY` (optional)
- ~30 GB disk space

**Session management:** This notebook runs too long for one Colab session.
- Run baselines first (cell 23), then AEIoU grid (cell 25) in subsequent sessions
- Drive backup after every run ensures no work is lost on disconnect
- Resume cell (cell 7) restores all completed runs at session start
"""), "cell-01-setup-md"))

# Cell 2: pip install
cells.append(code_cell(src(r"""
# --- Install pinned dependencies + fiftyone for COCO subset download
!pip install --upgrade pip -q
!pip install -U "ultralytics==8.4.9" "wandb==0.24.1" "fiftyone" -q
print("Dependencies installed.")
"""), "cell-02-install"))

# Cell 3: git clone
cells.append(code_cell(src(r"""
# --- Idempotent git clone
import os, sys

REPO_PATH = "/content/amorphous-yolo"
if not os.path.exists(f"{REPO_PATH}/.git"):
    print("Cloning amorphous-yolo...")
    os.system(f"git clone https://github.com/nipun-taneja/amorphous-yolo.git {REPO_PATH}")
else:
    print("Repo already present — skipping clone.")

if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

os.chdir(REPO_PATH)
print(f"Working directory: {os.getcwd()}")
"""), "cell-03-clone"))

# Cell 4: constants
cells.append(code_cell(src(r"""
# --- All experiment constants
import math, time
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path("/content/amorphous-yolo")

# ── Dataset roots ──────────────────────────────────────────────────────────────
DATASET_AMORPHOUS = PROJECT_DIR / "datasets" / "coco-amorphous"
DATASET_RIGID     = PROJECT_DIR / "datasets" / "coco-rigid"

# ── Experiment directories (separate to allow parallel Drive backups) ──────────
EXPERIMENTS_AMO  = PROJECT_DIR / "experiments_coco_amorphous"
EXPERIMENTS_RIG  = PROJECT_DIR / "experiments_coco_rigid"
ANALYSIS_DIR_AMO = EXPERIMENTS_AMO / "analysis"
ANALYSIS_DIR_RIG = EXPERIMENTS_RIG / "analysis"
MANIFEST_AMO     = EXPERIMENTS_AMO / "manifest.json"
MANIFEST_RIG     = EXPERIMENTS_RIG / "manifest.json"

for d in [EXPERIMENTS_AMO, EXPERIMENTS_RIG, ANALYSIS_DIR_AMO, ANALYSIS_DIR_RIG]:
    d.mkdir(parents=True, exist_ok=True)

# ── Google Drive ───────────────────────────────────────────────────────────────
DRIVE_ROOT       = Path("/content/drive/MyDrive/amorphous_yolo")
DRIVE_AMO        = DRIVE_ROOT / "experiments_coco_amorphous"
DRIVE_RIG        = DRIVE_ROOT / "experiments_coco_rigid"
DRIVE_AVAILABLE  = False

# ── Training hyper-parameters ─────────────────────────────────────────────────
EPOCHS   = 20
IMGSZ    = 640
DEVICE   = 0
MODEL_PT = "yolo26n.pt"
SEEDS    = [42]

# ── COCO amorphous categories (15) ────────────────────────────────────────────
# Deformable, non-rigid: animals + selected food items
AMORPHOUS_CATS = [
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
    "broccoli", "pizza", "cake", "potted plant", "teddy bear",
]

# ── COCO rigid categories (10) ────────────────────────────────────────────────
# Well-defined, precise boundaries: vehicles + electronics
RIGID_CATS = [
    "car", "bus", "truck",
    "chair", "dining table",
    "laptop", "tv", "microwave", "refrigerator", "cell phone",
]

# ── Loss configs ──────────────────────────────────────────────────────────────
BASELINE_LOSS_NAMES = ["iou", "giou", "diou", "ciou", "eiou", "eciou", "siou", "wiou"]
AEIOU_RIGIDITIES    = [round(x * 0.1, 1) for x in range(1, 11)]

def _fmt_r(r):
    return str(r).replace(".", "p")

ALL_LOSS_KEYS = BASELINE_LOSS_NAMES + [f"aeiou_r{_fmt_r(r)}" for r in AEIOU_RIGIDITIES]

# ── Amorphous split configs (3 splits: clean + low + high noise) ──────────────
SPLIT_CONFIGS_AMO = {
    "clean": PROJECT_DIR / "data" / "coco_amorphous.yaml",
    "low":   PROJECT_DIR / "data" / "coco_amorphous_low.yaml",
    "high":  PROJECT_DIR / "data" / "coco_amorphous_high.yaml",
}

# ── Rigid split config (clean only — control experiment, no noise needed) ─────
SPLIT_CONFIGS_RIG = {
    "clean": PROJECT_DIR / "data" / "coco_rigid.yaml",
}

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "iou":        "#888888", "giou":       "#BC6C25", "diou":       "#606C38",
    "ciou":       "#DDA15E", "eiou":       "#E63946", "eciou":      "#9B2226",
    "siou":       "#6A0572", "wiou":       "#FF6B6B",
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
    LOSS_LABELS[f"aeiou_r{_fmt_r(r)}"] = f"AEIoU \u03bb={r}"

n_amo = (len(BASELINE_LOSS_NAMES) + len(AEIOU_RIGIDITIES)) * len(SPLIT_CONFIGS_AMO) * len(SEEDS)
n_rig = (len(BASELINE_LOSS_NAMES) + len(AEIOU_RIGIDITIES)) * len(SPLIT_CONFIGS_RIG) * len(SEEDS)

print("Constants loaded.")
print(f"  Amorphous categories ({len(AMORPHOUS_CATS)}): {AMORPHOUS_CATS}")
print(f"  Rigid categories ({len(RIGID_CATS)}): {RIGID_CATS}")
print(f"  Amorphous runs: {n_amo}  (8 losses + 10 AEIoU) x 3 splits x 1 seed")
print(f"  Rigid runs:     {n_rig}  (8 losses + 10 AEIoU) x 1 split (clean only)")
print(f"  Total:          {n_amo + n_rig}")
"""), "cell-04-constants"))

# Cell 5: WandB
cells.append(code_cell(src(r"""
# --- WandB setup
import os, wandb

WANDB_PROJECT = "amorphous-yolo-coco"

try:
    from google.colab import userdata
    api_key = userdata.get("WANDB_API_KEY")
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
        print("WandB API key loaded from Colab secrets.")
except Exception:
    pass

if os.environ.get("WANDB_API_KEY"):
    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=False)
    print(f"WandB logged in. Project: {WANDB_PROJECT}")
else:
    os.environ["WANDB_MODE"] = "disabled"
    print("WANDB_API_KEY not found — WandB disabled.")
"""), "cell-05-wandb"))

# Cell 6: Drive markdown
cells.append(md_cell(src(r"""
### Google Drive: Mount, Restore & Persist

This notebook writes to **two** Drive directories:
- `MyDrive/amorphous_yolo/experiments_coco_amorphous/`
- `MyDrive/amorphous_yolo/experiments_coco_rigid/`

Results are saved after **every completed run**. On restart, run the restore cell
to recover all completed runs before resuming the training grid.
"""), "cell-06-drive-md"))

# Cell 7: Drive mount + restore
cells.append(code_cell(src(r"""
# --- Google Drive: mount + restore completed runs
import shutil

def mount_drive():
    global DRIVE_AVAILABLE
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        DRIVE_AMO.mkdir(parents=True, exist_ok=True)
        DRIVE_RIG.mkdir(parents=True, exist_ok=True)
        DRIVE_AVAILABLE = True
        print(f"Drive mounted.")
        print(f"  Amorphous: {DRIVE_AMO}")
        print(f"  Rigid:     {DRIVE_RIG}")
    except Exception as e:
        print(f"Drive not available ({e}). Running without Drive persistence.")
        DRIVE_AVAILABLE = False
    return DRIVE_AVAILABLE


def restore_from_drive(experiments_dir, drive_dir, label=""):
    if not DRIVE_AVAILABLE or not drive_dir.exists():
        print(f"  {label}: Nothing to restore.")
        return 0
    restored = 0
    for drive_run in sorted(drive_dir.iterdir()):
        if not drive_run.is_dir():
            continue
        local_run = experiments_dir / drive_run.name
        if (drive_run / "results.csv").exists() and not (local_run / "results.csv").exists():
            shutil.copytree(str(drive_run), str(local_run), dirs_exist_ok=True)
            restored += 1
            print(f"  [RESTORE] {drive_run.name}")
    if restored == 0:
        print(f"  {label}: Nothing to restore — local is up to date.")
    else:
        print(f"  {label}: Restored {restored} run(s) from Drive.")
    return restored


mount_drive()
print("\nRestoring amorphous runs...")
restore_from_drive(EXPERIMENTS_AMO, DRIVE_AMO, "amorphous")
print("\nRestoring rigid runs...")
restore_from_drive(EXPERIMENTS_RIG, DRIVE_RIG, "rigid")
print("\nRestore complete.")
"""), "cell-07-drive"))

# Cell 8: Dataset section markdown
cells.append(md_cell(src(r"""
## Section 2 · COCO 2017 Dataset Setup

### Amorphous subset (15 classes)
Animals and food items with deformable, non-rigid boundaries:
`bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe,
broccoli, pizza, cake, potted plant, teddy bear`

### Rigid subset (10 classes) — control
Objects with precise, well-defined boundaries:
`car, bus, truck, chair, dining table, laptop, tv, microwave, refrigerator, cell phone`

### Download strategy
We use `fiftyone` to download only the COCO images containing our target categories.
This avoids downloading the full 25 GB COCO dataset. Only ~5-8 GB is needed.

**Expected sizes:**
- Amorphous: ~30,000 train images / ~1,500 val images
- Rigid: ~25,000 train images / ~1,200 val images
"""), "cell-08-dataset-md"))

# Cell 9: Download COCO via fiftyone
cells.append(code_cell(src(r"""
# --- Download COCO 2017 subsets via fiftyone (idempotent)
# fiftyone downloads only images containing the requested categories.
# This is much faster than downloading the full COCO dataset (25 GB).
import fiftyone as fo
import fiftyone.zoo as foz
import os, shutil
from pathlib import Path

def _coco_to_yolo_format(fo_dataset, output_root: Path, split_name: str):
    # Convert fiftyone dataset detections to YOLO label files.
    # fiftyone bboxes are [x, y, w, h] relative (top-left + size, range [0,1]).
    # YOLO format: class_id cx cy w h (all relative, cx/cy = center).
    img_dir = output_root / split_name / "images"
    lbl_dir = output_root / split_name / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Build class-name to class-id mapping from dataset info
    class_names = fo_dataset.default_classes or fo_dataset.distinct("ground_truth.detections.label")
    class_to_id = {name: i for i, name in enumerate(sorted(class_names))}

    written = 0
    skipped = 0
    for sample in fo_dataset.iter_samples():
        src_path = Path(sample.filepath)
        if not src_path.exists():
            skipped += 1
            continue

        dets = sample.ground_truth.detections if sample.ground_truth else []
        if not dets:
            skipped += 1
            continue

        lines = []
        for det in dets:
            label = det.label
            if label not in class_to_id:
                continue
            cls_id = class_to_id[label]
            # fiftyone bbox: [x_min, y_min, width, height] normalised
            bx, by, bw, bh = det.bounding_box
            cx = bx + bw / 2
            cy = by + bh / 2
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if not lines:
            skipped += 1
            continue

        dst_img = img_dir / src_path.name
        if not dst_img.exists():
            shutil.copy2(src_path, dst_img)
        (lbl_dir / f"{src_path.stem}.txt").write_text("\n".join(lines) + "\n")
        written += 1

    print(f"  {split_name}: {written} images written, {skipped} skipped.")
    return written


def download_coco_subset(categories, output_root: Path, label: str):
    # Download COCO subset using fiftyone zoo, convert to YOLO format.
    output_root.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    n_train = len(list((output_root / "train" / "images").glob("*"))) if (output_root / "train" / "images").exists() else 0
    n_val   = len(list((output_root / "valid" / "images").glob("*"))) if (output_root / "valid" / "images").exists() else 0

    if n_train > 100 and n_val > 50:
        print(f"  {label}: already present (train={n_train}, val={n_val}) — skipping download.")
        return

    print(f"  Downloading COCO {label} subset ({len(categories)} categories)...")
    print(f"  Categories: {categories}")

    for fo_split, yolo_split in [("train", "train"), ("validation", "valid")]:
        print(f"  Processing {fo_split}...")
        try:
            ds = foz.load_zoo_dataset(
                "coco-2017",
                split=fo_split,
                label_types=["detections"],
                classes=categories,
                only_matching=True,
                dataset_name=f"coco_{label}_{fo_split}_tmp",
                overwrite=True,
            )
            _coco_to_yolo_format(ds, output_root, yolo_split)
            fo.delete_dataset(f"coco_{label}_{fo_split}_tmp")
        except Exception as e:
            print(f"  [WARN] {fo_split} download failed: {e}")
            raise

    print(f"  {label}: download and conversion complete.")


print("Downloading COCO subsets (this may take 10-20 minutes)...")
download_coco_subset(AMORPHOUS_CATS, DATASET_AMORPHOUS, "amorphous")
print()
download_coco_subset(RIGID_CATS, DATASET_RIGID, "rigid")

# Summary
for label, root in [("Amorphous", DATASET_AMORPHOUS), ("Rigid", DATASET_RIGID)]:
    n_tr = len(list((root / "train" / "images").glob("*")))
    n_va = len(list((root / "valid" / "images").glob("*")))
    print(f"  {label}: train={n_tr}  val={n_va}")
"""), "cell-09-download"))

# Cell 10: Noise splits for amorphous
cells.append(code_cell(src(r"""
# --- Build noise-perturbed validation splits for COCO amorphous (idempotent)
# Only amorphous needs noise splits (clean, low, high).
# Rigid subset uses clean validation only (control experiment).
import numpy as np
import shutil

SIGMA_LOW  = 0.02
SIGMA_HIGH = 0.08


def _jitter_label_file(src_lbl, dst_lbl, sigma, rng):
    lines = src_lbl.read_text().strip().split("\n")
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
    dst_lbl.write_text("\n".join(out) + "\n")


def build_noise_splits(root: Path, seed: int = 42):
    src_img = root / "valid" / "images"
    src_lbl = root / "valid" / "labels"
    n_valid = len(list(src_img.glob("*")))
    rng = np.random.default_rng(seed)

    for split_name, sigma in [("valid_low", SIGMA_LOW), ("valid_high", SIGMA_HIGH)]:
        dst_img = root / split_name / "images"
        dst_lbl = root / split_name / "labels"

        if dst_img.exists() and len(list(dst_img.glob("*"))) >= n_valid - 5:
            print(f"  {split_name}: already exists — skipping.")
            continue

        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        n_processed = 0
        for img_path in sorted(src_img.glob("*")):
            stem = img_path.stem
            dst = dst_img / img_path.name
            if not dst.exists():
                try:
                    dst.symlink_to(img_path)
                except (OSError, NotImplementedError):
                    shutil.copy2(img_path, dst)
            sl = src_lbl / f"{stem}.txt"
            dl = dst_lbl / f"{stem}.txt"
            if sl.exists():
                _jitter_label_file(sl, dl, sigma, rng)
                n_processed += 1

        print(f"  {split_name}: {n_processed} labels jittered (\u03c3={sigma})")


print("Building noise splits for COCO amorphous...")
build_noise_splits(DATASET_AMORPHOUS)
print("Noise splits ready.")
print("\nNote: COCO rigid uses clean validation only (no noise splits needed for control).")
"""), "cell-10-noise"))

# Cell 11: Verify
cells.append(code_cell(src(r"""
# --- Verify all split configs exist and have correct image counts
import yaml

print("=== COCO Amorphous splits ===")
print(f"{'Split':<10} {'YAML':<44} {'Val images':>12}")
print("-" * 69)
for split_name, cfg_path in SPLIT_CONFIGS_AMO.items():
    assert cfg_path.exists(), f"Missing yaml: {cfg_path}"
    cfg = yaml.safe_load(cfg_path.read_text())
    val_dir = DATASET_AMORPHOUS / cfg["val"]
    n_imgs = len(list(val_dir.glob("*")))
    status = "\u2713" if n_imgs >= 100 else "\u2717 CHECK"
    print(f"{split_name:<10} {str(cfg_path.name):<44} {n_imgs:>10}  {status}")
n_tr_amo = len(list((DATASET_AMORPHOUS / "train" / "images").glob("*")))
print(f"Train images (amorphous): {n_tr_amo}")

print("\n=== COCO Rigid splits ===")
for split_name, cfg_path in SPLIT_CONFIGS_RIG.items():
    assert cfg_path.exists(), f"Missing yaml: {cfg_path}"
    cfg = yaml.safe_load(cfg_path.read_text())
    val_dir = DATASET_RIGID / cfg["val"]
    n_imgs = len(list(val_dir.glob("*")))
    status = "\u2713" if n_imgs >= 100 else "\u2717 CHECK"
    print(f"{split_name:<10} {str(cfg_path.name):<44} {n_imgs:>10}  {status}")
n_tr_rig = len(list((DATASET_RIGID / "train" / "images").glob("*")))
print(f"Train images (rigid): {n_tr_rig}")

print("\nAll splits verified.")
"""), "cell-11-verify"))

# Cell 12: Sample visualisation (amorphous)
cells.append(code_cell(src(r"""
# --- Visualise 5 random COCO amorphous training images
import cv2, random
import numpy as np
import matplotlib.pyplot as plt

def _show_samples(dataset_root, title, n=5, seed=77):
    img_dir = dataset_root / "train" / "images"
    lbl_dir = dataset_root / "train" / "labels"
    all_imgs = sorted(img_dir.glob("*"))
    if not all_imgs:
        print(f"No images found in {img_dir}")
        return

    random.seed(seed)
    sample = random.sample(all_imgs, min(n, len(all_imgs)))

    fig, axes = plt.subplots(1, len(sample), figsize=(4 * len(sample), 4))
    if len(sample) == 1:
        axes = [axes]

    for ax, img_path in zip(axes, sample):
        img = cv2.imread(str(img_path))
        if img is None:
            ax.axis("off")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    _, cx, cy, w, h = [float(v) for v in parts[:5]]
                    x1 = int((cx - w / 2) * W)
                    y1 = int((cy - h / 2) * H)
                    x2 = int((cx + w / 2) * W)
                    y2 = int((cy + h / 2) * H)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 80, 80), 2)
        ax.imshow(img)
        ax.set_title(img_path.stem[:18], fontsize=7)
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    return fig


fig_amo = _show_samples(DATASET_AMORPHOUS, "COCO Amorphous — Sample Training Images")
if fig_amo:
    fig_amo.savefig(str(ANALYSIS_DIR_AMO / "fig0_coco_amorphous_samples.png"), dpi=100, bbox_inches="tight")
    plt.show()
    print("Amorphous samples saved.")
"""), "cell-12-visualize"))

# Cell 13: Theory markdown
cells.append(md_cell(src(r"""
## Section 3 · Theory & Experimental Design

### AEIoU Hypothesis for COCO

**Amorphous subset (animals + food):**
- Animal bounding boxes are annotation-dependent — different annotators draw different extents
- Food items (pizza, cake) have irregular, blob-like shapes
- **Prediction:** AEIoU λ=0.3–0.5 > EIoU (same as Kvasir and ISIC)

**Rigid subset (vehicles + electronics) — control:**
- Cars, laptops, TVs have crisp, well-defined rectangular boundaries
- Labels are precise and annotation-independent
- **Prediction:** AEIoU δ ≈ 0 (no benefit from reducing λ); at λ=1.0 performance matches EIoU

### The Key Comparison
If the delta AEIoU − EIoU is significantly larger on amorphous than rigid:
→ This confirms AEIoU specifically targets annotation noise, not just general metric improvement

See **notebook 07** for the cross-dataset delta scatter plot.
"""), "cell-13-theory-md"))

# Cell 14: Monkey-patch markdown
cells.append(md_cell(src(r"""
## Section 4 · Monkey-Patch Infrastructure

Identical to notebooks 03 and 04. Verbatim copy.
"""), "cell-14-patch-md"))

# Cell 15: Monkey-patch code
cells.append(code_cell(src(r"""
# --- Full monkey-patch implementation (verbatim from notebook 03)
import types
import torch
import torch.nn.functional as F
import ultralytics.utils.loss as loss_mod

_ORIGINAL_BBOX_FORWARD = loss_mod.BboxLoss.forward


def _make_bbox_forward(loss_fn_instance):
    def bbox_loss_forward(
        self, pred_dist, pred_bboxes, anchor_points,
        target_bboxes, target_scores, target_scores_sum,
        fg_mask, imgsz, stride,
    ):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        per_box = loss_fn_instance(pred_bboxes[fg_mask], target_bboxes[fg_mask])
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
    suffix = f"(\u03bb={loss_fn_instance.rigidity})" if hasattr(loss_fn_instance, "rigidity") else ""
    print(f"  [PATCH] BboxLoss.forward \u2192 {type(loss_fn_instance).__name__}{suffix}")


def restore_loss():
    loss_mod.BboxLoss.forward = _ORIGINAL_BBOX_FORWARD


print("Patch infrastructure ready.")
"""), "cell-15-patch"))

# Cell 16: Patch verify
cells.append(code_cell(src(r"""
# --- Verify patch round-trip
from src.losses import EIoULoss
_t = EIoULoss(reduction="none")
patch_loss(_t)
assert loss_mod.BboxLoss.forward is not _ORIGINAL_BBOX_FORWARD
restore_loss()
assert loss_mod.BboxLoss.forward is _ORIGINAL_BBOX_FORWARD
print("Patch round-trip verified.")
"""), "cell-16-patch-verify"))

# Cell 17: Loss registry markdown
cells.append(md_cell(src(r"""
## Section 5 · Loss Registry

Identical to notebooks 03 and 04. Both COCO subsets use the same 18-loss registry.
"""), "cell-17-registry-md"))

# Cell 18: Loss registry
cells.append(code_cell(src(r"""
# --- Loss registry
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

print(f"Baselines ({len(BASELINE_LOSS_REGISTRY)}): {list(BASELINE_LOSS_REGISTRY)}")
print(f"AEIoU ({len(AEIOU_LOSS_REGISTRY)}): {list(AEIOU_LOSS_REGISTRY)}")
"""), "cell-18-registry"))

# Cell 19: Training infrastructure markdown
cells.append(md_cell(src(r"""
## Section 6 · Training Infrastructure

### Run naming convention
```
coco_amorphous_yolo26n_{loss}_{split}_s{seed}_e{epochs}
coco_rigid_yolo26n_{loss}_{split}_s{seed}_e{epochs}

coco_amorphous_yolo26n_eiou_clean_s42_e20
coco_amorphous_yolo26n_aeiou_r0p3_low_s42_e20
coco_rigid_yolo26n_eiou_clean_s42_e20
```

Two separate `run_training()` functions: one for amorphous, one for rigid.
They differ only in `run_prefix`, `experiments_dir`, and `drive_dir`.
"""), "cell-19-training-md"))

# Cell 20: run_training (shared factory)
cells.append(code_cell(src(r"""
# --- Training function factory: creates run_training for amorphous or rigid subset
import json, shutil
from pathlib import Path as _Path
from datetime import datetime
from ultralytics import YOLO


def _load_manifest(manifest_path):
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {}


def _write_manifest(manifest_path, run_name, meta):
    manifest = _load_manifest(manifest_path)
    manifest[run_name] = meta
    manifest_path.write_text(json.dumps(manifest, indent=2))


def _sync_to_drive(run_name, experiments_dir, drive_dir):
    if not DRIVE_AVAILABLE:
        return
    local_run = experiments_dir / run_name
    drive_run = drive_dir / run_name
    try:
        shutil.copytree(str(local_run), str(drive_run), dirs_exist_ok=True)
        print(f"  [DRIVE] Synced {run_name}")
    except Exception as e:
        print(f"  [DRIVE] Sync failed for {run_name}: {e}")


def _make_epoch_cb(run_name, drive_dir):
    def _on_epoch_end(trainer):
        if not DRIVE_AVAILABLE:
            return
        last_pt = _Path(trainer.save_dir) / "weights" / "last.pt"
        if not last_pt.exists():
            return
        dw = drive_dir / run_name / "weights"
        dw.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(str(last_pt), str(dw / "last.pt"))
        except Exception:
            pass
    return _on_epoch_end


def make_run_training(run_prefix, experiments_dir, drive_dir, manifest_path):
    # Returns a run_training() function bound to the given dataset context.
    def run_training(loss_name, loss_fn, split_name, yaml_path,
                     seed=42, epochs=None, imgsz=None, device=None):
        epochs = epochs if epochs is not None else EPOCHS
        imgsz  = imgsz  if imgsz  is not None else IMGSZ
        device = device if device is not None else DEVICE

        run_name = f"{run_prefix}_yolo26n_{loss_name}_{split_name}_s{seed}_e{epochs}"
        run_dir  = experiments_dir / run_name

        if (run_dir / "results.csv").exists():
            print(f"[SKIP] {run_name}")
            return run_dir

        drive_last_pt = drive_dir / run_name / "weights" / "last.pt"
        resuming = DRIVE_AVAILABLE and drive_last_pt.exists()

        if resuming:
            print(f"\n{'='*70}\n[RESUME] {run_name}\n{'='*70}")
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "weights").mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(drive_last_pt), str(run_dir / "weights" / "last.pt"))
        else:
            print(f"\n{'='*70}")
            print(f"[START] {run_name}")
            print(f"  loss={loss_name}  split={split_name}  seed={seed}  epochs={epochs}")
            print(f"{'='*70}")

        meta = {
            "loss": loss_name, "split": split_name, "seed": seed, "epochs": epochs,
            "rigidity": float(getattr(loss_fn, "rigidity", -1) or -1),
            "run_dir": str(run_dir), "weights": str(run_dir / "weights" / "best.pt"),
            "results_csv": str(run_dir / "results.csv"),
            "timestamp": datetime.now().isoformat(),
            "status": "running", "resumed": resuming,
        }
        _write_manifest(manifest_path, run_name, meta)

        t_start = time.time()
        try:
            import os as _os
            _os.environ["WANDB_PROJECT"] = WANDB_PROJECT
            _os.environ["WANDB_NAME"]    = run_name
            _os.environ["WANDB_TAGS"]    = f"{loss_name},{split_name},{run_prefix}"

            patch_loss(loss_fn)

            if resuming:
                model = YOLO(str(run_dir / "weights" / "last.pt"))
                model.add_callback("on_train_epoch_end", _make_epoch_cb(run_name, drive_dir))
                results = model.train(resume=True)
            else:
                model = YOLO(MODEL_PT)
                model.add_callback("on_train_epoch_end", _make_epoch_cb(run_name, drive_dir))
                results = model.train(
                    data=str(yaml_path), epochs=epochs, imgsz=imgsz,
                    project=str(experiments_dir), name=run_name,
                    device=device, seed=seed, exist_ok=True,
                )

            try:
                (run_dir / "run_meta.json").write_text(json.dumps(results.results_dict, indent=2))
            except Exception as e:
                print(f"  [WARN] run_meta.json: {e}")

            meta["status"] = "complete"
            meta["elapsed_sec"] = round(time.time() - t_start, 1)

            try:
                import wandb as _wandb
                if _wandb.run is not None:
                    _wandb.finish()
            except Exception:
                pass

        except Exception as e:
            print(f"  [ERROR] {run_name} failed: {e}")
            meta["status"] = "failed"
            meta["error"] = str(e)
            raise

        finally:
            restore_loss()
            _write_manifest(manifest_path, run_name, meta)
            _sync_to_drive(run_name, experiments_dir, drive_dir)

        print(f"[DONE] {run_name}")
        return run_dir

    return run_training


# Create bound run_training functions for each subset
run_training_amo = make_run_training("coco_amorphous", EXPERIMENTS_AMO, DRIVE_AMO, MANIFEST_AMO)
run_training_rig = make_run_training("coco_rigid", EXPERIMENTS_RIG, DRIVE_RIG, MANIFEST_RIG)

print("run_training_amo() and run_training_rig() ready.")
"""), "cell-20-run-training"))

# Cell 21: Amorphous baselines markdown
cells.append(md_cell(src(r"""
## Section 7 · Amorphous Subset — Baseline Training (8 losses × 3 splits)

24 runs total. These are the longest runs (large COCO subset).
Completed runs are skipped on re-run.
"""), "cell-21-amo-baseline-md"))

# Cell 22: Amorphous baselines
cells.append(code_cell(src(r"""
# --- COCO amorphous baseline training: 8 losses x 3 splits x N seeds
for loss_name, loss_fn in BASELINE_LOSS_REGISTRY.items():
    for split_name, cfg_path in SPLIT_CONFIGS_AMO.items():
        for seed in SEEDS:
            run_training_amo(loss_name=loss_name, loss_fn=loss_fn,
                             split_name=split_name, yaml_path=cfg_path, seed=seed)

restore_loss()
n_base = len(BASELINE_LOSS_REGISTRY) * len(SPLIT_CONFIGS_AMO) * len(SEEDS)
print(f"\nAmorphous baseline: {n_base} runs complete (or skipped).")
"""), "cell-22-amo-baseline-train"))

# Cell 23: Amorphous AEIoU markdown
cells.append(md_cell(src(r"""
## Section 8 · Amorphous Subset — AEIoU Grid (10 λ × 3 splits)

30 runs. **Expected:** peak mAP at λ ≈ 0.3–0.5 (same as Kvasir, ISIC).
"""), "cell-23-amo-aeiou-md"))

# Cell 24: Amorphous AEIoU training
cells.append(code_cell(src(r"""
# --- COCO amorphous AEIoU grid: 10 lambdas x 3 splits x N seeds
total = len(AEIOU_RIGIDITIES) * len(SPLIT_CONFIGS_AMO) * len(SEEDS)
done  = 0

for r in AEIOU_RIGIDITIES:
    loss_name = f"aeiou_r{_fmt_r(r)}"
    loss_fn   = AEIOU_LOSS_REGISTRY[loss_name]
    for split_name, cfg_path in SPLIT_CONFIGS_AMO.items():
        for seed in SEEDS:
            done += 1
            print(f"\n[{done}/{total}] lam={r}  split={split_name}  seed={seed}")
            run_training_amo(loss_name=loss_name, loss_fn=loss_fn,
                             split_name=split_name, yaml_path=cfg_path, seed=seed)

restore_loss()
print(f"\nAmorphous AEIoU grid: {total} runs complete (or skipped).")
"""), "cell-24-amo-aeiou-train"))

# Cell 25: Rigid baselines markdown
cells.append(md_cell(src(r"""
## Section 9 · Rigid Subset — Baseline Training (8 losses × 1 split)

8 runs (clean split only — rigid objects don't need noise splits for this paper).
**Expected:** All losses perform similarly; AEIoU delta ≈ 0.
"""), "cell-25-rig-baseline-md"))

# Cell 26: Rigid baselines
cells.append(code_cell(src(r"""
# --- COCO rigid baseline training: 8 losses x 1 split (clean only) x N seeds
for loss_name, loss_fn in BASELINE_LOSS_REGISTRY.items():
    for split_name, cfg_path in SPLIT_CONFIGS_RIG.items():
        for seed in SEEDS:
            run_training_rig(loss_name=loss_name, loss_fn=loss_fn,
                             split_name=split_name, yaml_path=cfg_path, seed=seed)

restore_loss()
n_base = len(BASELINE_LOSS_REGISTRY) * len(SPLIT_CONFIGS_RIG) * len(SEEDS)
print(f"\nRigid baseline: {n_base} runs complete (or skipped).")
"""), "cell-26-rig-baseline-train"))

# Cell 27: Rigid AEIoU markdown
cells.append(md_cell(src(r"""
## Section 10 · Rigid Subset — AEIoU Grid (10 λ × 1 split)

10 runs (clean split only). These are the control runs.
**Expected:** AEIoU δ ≈ 0 across all λ values (rigid labels are precise).
"""), "cell-27-rig-aeiou-md"))

# Cell 28: Rigid AEIoU training
cells.append(code_cell(src(r"""
# --- COCO rigid AEIoU grid: 10 lambdas x 1 split (clean only) x N seeds
total = len(AEIOU_RIGIDITIES) * len(SPLIT_CONFIGS_RIG) * len(SEEDS)
done  = 0

for r in AEIOU_RIGIDITIES:
    loss_name = f"aeiou_r{_fmt_r(r)}"
    loss_fn   = AEIOU_LOSS_REGISTRY[loss_name]
    for split_name, cfg_path in SPLIT_CONFIGS_RIG.items():
        for seed in SEEDS:
            done += 1
            print(f"\n[{done}/{total}] lam={r}  split={split_name}  seed={seed}")
            run_training_rig(loss_name=loss_name, loss_fn=loss_fn,
                             split_name=split_name, yaml_path=cfg_path, seed=seed)

restore_loss()
print(f"\nRigid AEIoU grid: {total} runs complete (or skipped).")
"""), "cell-28-rig-aeiou-train"))

# Cell 29: Results loading markdown
cells.append(md_cell(src(r"""
## Section 11 · Results Collection

Load results from both subsets into separate DataFrames.
"""), "cell-29-results-md"))

# Cell 30: Load results
cells.append(code_cell(src(r"""
# --- Load results for both subsets
import pandas as pd

def load_results(run_prefix, experiments_dir, split_configs, label):
    cache = experiments_dir / "all_results_combined.csv"
    if cache.exists():
        print(f"  {label}: loading from cache")
        return pd.read_csv(cache)

    print(f"  {label}: building from CSVs...")
    dfs = []
    for loss_name in ALL_LOSS_KEYS:
        for split_name in split_configs:
            for seed in SEEDS:
                run_name = f"{run_prefix}_yolo26n_{loss_name}_{split_name}_s{seed}_e{EPOCHS}"
                csv_path = experiments_dir / run_name / "results.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    df.columns = df.columns.str.strip()
                    df["run_name"] = run_name
                    df["loss"] = loss_name
                    df["split"] = split_name
                    df["seed"] = seed
                    df["epoch"] = df.index + 1
                    dfs.append(df)
                else:
                    print(f"    [MISSING] {run_name}")
    if not dfs:
        print(f"  {label}: no results found.")
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(cache, index=False)
    print(f"  {label}: {len(dfs)} runs loaded.")
    return combined


print("Loading results...")
df_amo = load_results("coco_amorphous", EXPERIMENTS_AMO, SPLIT_CONFIGS_AMO, "Amorphous")
df_rig = load_results("coco_rigid",     EXPERIMENTS_RIG, SPLIT_CONFIGS_RIG, "Rigid")

print(f"\nAmorphous DataFrame: {df_amo.shape}")
print(f"Rigid DataFrame:     {df_rig.shape}")
"""), "cell-30-load-results"))

# Cell 31: Summary table
cells.append(code_cell(src(r"""
# --- Build summary tables for both subsets
import pandas as pd
import numpy as np

MAP50_COL = "metrics/mAP50(B)"
MAP95_COL = "metrics/mAP50-95(B)"


def build_summary(df, split_configs, label, analysis_dir):
    if df.empty:
        print(f"  {label}: no data.")
        return pd.DataFrame()

    df_final = df.groupby("run_name").last().reset_index()
    rows = []
    for loss_name in ALL_LOSS_KEYS:
        for split in split_configs:
            sub = df_final[(df_final["loss"] == loss_name) & (df_final["split"] == split)]
            if sub.empty:
                continue
            row = {"loss": loss_name, "split": split, "label": LOSS_LABELS.get(loss_name, loss_name)}
            for col, tag in [(MAP95_COL, "map95"), (MAP50_COL, "map50")]:
                if col in sub.columns:
                    row[f"{tag}_mean"] = sub[col].mean()
                    row[f"{tag}_std"]  = sub[col].std() if len(sub) > 1 else 0.0
            rows.append(row)

    df_sum = pd.DataFrame(rows)
    df_sum.to_csv(analysis_dir / "summary_table.csv", index=False)

    if "map95_mean" in df_sum.columns:
        clean = df_sum[df_sum["split"] == "clean"].sort_values("map95_mean", ascending=False)
        print(f"\n=== COCO {label} — Clean Split mAP50-95 Ranking ===")
        print(f"{'Loss':<22} {'mAP50-95':>10} {'mAP50':>10}")
        print("-" * 45)
        for _, row in clean.iterrows():
            m95 = f"{row.get('map95_mean', float('nan')):.4f}"
            m50 = f"{row.get('map50_mean', float('nan')):.4f}"
            print(f"{row['label']:<22} {m95:>10} {m50:>10}")
    return df_sum


df_sum_amo = build_summary(df_amo, SPLIT_CONFIGS_AMO, "Amorphous", ANALYSIS_DIR_AMO)
df_sum_rig = build_summary(df_rig, SPLIT_CONFIGS_RIG, "Rigid",     ANALYSIS_DIR_RIG)
"""), "cell-31-summary"))

# Cell 32: Key plots markdown
cells.append(md_cell(src(r"""
## Section 12 · Core Analysis: Amorphous vs Rigid Comparison

The central comparison: AEIoU delta on amorphous vs rigid objects.
"""), "cell-32-analysis-md"))

# Cell 33: Lambda curves both subsets
cells.append(code_cell(src(r"""
# --- Fig 3: Lambda-vs-mAP for both subsets (side by side — the key figure)
import matplotlib.pyplot as plt

MAP95_COL = "metrics/mAP50-95(B)"

fig, (ax_amo, ax_rig) = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

for ax, df, split_configs, title in [
    (ax_amo, df_amo, SPLIT_CONFIGS_AMO, "COCO Amorphous (15 classes)"),
    (ax_rig, df_rig, SPLIT_CONFIGS_RIG, "COCO Rigid — Control (10 classes)"),
]:
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        continue

    df_final = df.groupby("run_name").last().reset_index()

    for split, ls in [("clean", "-"), ("low", "--"), ("high", ":")]:
        if split not in split_configs:
            continue
        lambdas, maps = [], []
        for r in AEIOU_RIGIDITIES:
            k = f"aeiou_r{_fmt_r(r)}"
            sub = df_final[(df_final["loss"] == k) & (df_final["split"] == split)]
            if not sub.empty and MAP95_COL in sub.columns:
                lambdas.append(r)
                maps.append(sub[MAP95_COL].mean())
        if lambdas:
            ax.plot(lambdas, maps, marker="o", linestyle=ls, linewidth=2,
                    label=f"AEIoU ({split})")

    for ref in ["eiou", "ciou"]:
        sub = df_final[(df_final["loss"] == ref) & (df_final["split"] == "clean")]
        if not sub.empty and MAP95_COL in sub.columns:
            v = sub[MAP95_COL].mean()
            ax.axhline(v, linestyle="--", color=PALETTE[ref], alpha=0.7,
                       label=f"{LOSS_LABELS[ref]} (clean) = {v:.4f}")

    ax.set_xlabel("\u03bb (AEIoU rigidity)")
    ax.set_ylabel("mAP50-95")
    ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle("COCO: AEIoU \u03bb-vs-mAP — Amorphous vs Rigid Control", fontsize=13)
plt.tight_layout()
fig.savefig(str(ANALYSIS_DIR_AMO / "fig3_lambda_curve_amorphous_vs_rigid.png"),
            dpi=150, bbox_inches="tight")
plt.show()
print("Fig 3 saved.")
"""), "cell-33-fig3"))

# Cell 34: AEIoU delta comparison
cells.append(code_cell(src(r"""
# --- Fig 4: AEIoU delta (best AEIoU - EIoU) for amorphous vs rigid
# This is the central evidence figure: large delta on amorphous, ~0 on rigid.
import matplotlib.pyplot as plt
import numpy as np

MAP95_COL = "metrics/mAP50-95(B)"

deltas = {"amorphous": {}, "rigid": {}}

for label, df, experiments_dir in [("amorphous", df_amo, EXPERIMENTS_AMO),
                                     ("rigid",     df_rig, EXPERIMENTS_RIG)]:
    if df.empty:
        continue
    df_final = df.groupby("run_name").last().reset_index()
    eiou_sub = df_final[(df_final["loss"] == "eiou") & (df_final["split"] == "clean")]
    eiou_map = eiou_sub[MAP95_COL].mean() if not eiou_sub.empty and MAP95_COL in eiou_sub.columns else float("nan")

    for r in AEIOU_RIGIDITIES:
        k = f"aeiou_r{_fmt_r(r)}"
        sub = df_final[(df_final["loss"] == k) & (df_final["split"] == "clean")]
        if not sub.empty and MAP95_COL in sub.columns:
            deltas[label][r] = sub[MAP95_COL].mean() - eiou_map

if deltas["amorphous"] or deltas["rigid"]:
    fig, ax = plt.subplots(figsize=(10, 6))
    lambdas = AEIOU_RIGIDITIES

    if deltas["amorphous"]:
        vals_amo = [deltas["amorphous"].get(r, float("nan")) for r in lambdas]
        ax.plot(lambdas, vals_amo, "o-", color="#00B4D8", linewidth=2.5, label="Amorphous \u0394")
    if deltas["rigid"]:
        vals_rig = [deltas["rigid"].get(r, float("nan")) for r in lambdas]
        ax.plot(lambdas, vals_rig, "s--", color="#E63946", linewidth=2.5, label="Rigid \u0394 (control)")

    ax.axhline(0, color="black", linewidth=1, linestyle="-")
    ax.fill_between(lambdas,
                    [deltas["amorphous"].get(r, 0) for r in lambdas],
                    [0] * len(lambdas),
                    alpha=0.15, color="#00B4D8")
    ax.set_xlabel("\u03bb (AEIoU rigidity)")
    ax.set_ylabel("mAP50-95 \u0394 vs EIoU (clean split)")
    ax.set_title("COCO: AEIoU Benefit — Amorphous vs Rigid Control\n"
                 "Positive \u0394 = AEIoU better than EIoU")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(ANALYSIS_DIR_AMO / "fig4_aeiou_delta_amorphous_vs_rigid.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Fig 4 saved.")
    print("\nKey deltas (best lambda):")
    for label in ["amorphous", "rigid"]:
        if deltas[label]:
            best_r = max(deltas[label], key=lambda r: deltas[label][r])
            print(f"  {label}: best \u03bb={best_r}  \u0394={deltas[label][best_r]:+.4f}")
"""), "cell-34-delta"))

# Cell 35: Noise robustness (amorphous)
cells.append(code_cell(src(r"""
# --- Fig 7: Noise robustness gap (amorphous subset only)
import matplotlib.pyplot as plt

if df_amo.empty:
    print("No amorphous results.")
else:
    MAP95_COL = "metrics/mAP50-95(B)"
    df_final = df_amo.groupby("run_name").last().reset_index()

    gaps = []
    for k in ALL_LOSS_KEYS:
        c = df_final[(df_final["loss"] == k) & (df_final["split"] == "clean")]
        h = df_final[(df_final["loss"] == k) & (df_final["split"] == "high")]
        if not c.empty and not h.empty and MAP95_COL in c.columns:
            gaps.append((k, c[MAP95_COL].mean() - h[MAP95_COL].mean()))

    if gaps:
        gaps.sort(key=lambda x: x[1])
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.barh([LOSS_LABELS.get(g[0], g[0]) for g in gaps],
                [g[1] for g in gaps],
                color=[PALETTE.get(g[0], "#999") for g in gaps], alpha=0.85)
        ax.set_xlabel("mAP50-95 drop (clean \u2212 high noise)")
        ax.set_title("COCO Amorphous — Noise Robustness Gap (smaller = more robust)")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        fig.savefig(str(ANALYSIS_DIR_AMO / "fig7_noise_robustness.png"), dpi=150, bbox_inches="tight")
        plt.show()
        print("Fig 7 saved.")
"""), "cell-35-fig7"))

# Cell 36: Summary markdown
cells.append(md_cell(src(r"""
## Section 13 · Summary & Conclusions

### Key Findings (COCO Amorphous vs Rigid)

| Metric | Expected |
|---|---|
| Amorphous lambda curve | Peak at λ ≈ 0.3–0.5 |
| Rigid lambda curve | Flat (delta ≈ 0 across all λ) |
| Fig 4 amorphous delta | Large positive delta |
| Fig 4 rigid delta | Near-zero delta |

### Cross-Dataset Confirmation
Consistent optimal λ across Kvasir (notebook 03), ISIC (notebook 04), and COCO amorphous
(this notebook) confirms the amorphous-boundary hypothesis.

Results feed into **notebook 07** (cross-dataset analysis) via:
- `experiments_coco_amorphous/` on Drive
- `experiments_coco_rigid/` on Drive
"""), "cell-36-summary-md"))

# Cell 37: Save artifacts
cells.append(code_cell(src(r"""
# --- Save all artifacts
import json

for label, experiments_dir, analysis_dir, manifest_path in [
    ("Amorphous", EXPERIMENTS_AMO, ANALYSIS_DIR_AMO, MANIFEST_AMO),
    ("Rigid",     EXPERIMENTS_RIG, ANALYSIS_DIR_RIG, MANIFEST_RIG),
]:
    print(f"\n=== COCO {label} ===")
    print(f"  Experiments: {experiments_dir}")
    figs = list(analysis_dir.glob("*.png"))
    tabs = list(analysis_dir.glob("*.csv"))
    print(f"  Figures: {len(figs)}")
    for f in sorted(figs):
        print(f"    {f.name:<50} {f.stat().st_size/1024:>6.1f} KB")
    print(f"  Tables: {len(tabs)}")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        complete = sum(1 for v in manifest.values() if v.get("status") == "complete")
        failed   = sum(1 for v in manifest.values() if v.get("status") == "failed")
        print(f"  Manifest: {len(manifest)} total | {complete} complete | {failed} failed")
"""), "cell-37-artifacts"))

# Cell 38: Final sync
cells.append(code_cell(src(r"""
# --- Final sync to Drive (both subsets)
import shutil

for label, experiments_dir, drive_dir, analysis_dir in [
    ("Amorphous", EXPERIMENTS_AMO, DRIVE_AMO, ANALYSIS_DIR_AMO),
    ("Rigid",     EXPERIMENTS_RIG, DRIVE_RIG, ANALYSIS_DIR_RIG),
]:
    if not DRIVE_AVAILABLE:
        print(f"  {label}: Drive not mounted — skipping.")
        continue

    drive_analysis = drive_dir / "analysis"
    drive_analysis.mkdir(parents=True, exist_ok=True)
    if analysis_dir.exists():
        shutil.copytree(str(analysis_dir), str(drive_analysis), dirs_exist_ok=True)
        print(f"  {label}: {len(list(drive_analysis.glob('*.png')))} PNGs synced")

    for fname in ["manifest.json", "all_results_combined.csv"]:
        src_path = experiments_dir / fname
        if src_path.exists():
            shutil.copy2(str(src_path), str(drive_dir / fname))

    print(f"  {label} backed up to {drive_dir}")

if not DRIVE_AVAILABLE:
    print("Drive not mounted. Run mount_drive() and re-run to back up.")
"""), "cell-38-sync"))

# Build notebook
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": [], "gpuType": "A100"},
        "accelerator": "GPU"
    },
    "cells": cells
}

out_path = "C:/Users/PINCstudent/Downloads/SFSU/899/project/amorphous-yolo/notebooks/05_coco_amorphous_rigid.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {out_path}")
print(f"Total cells: {len(cells)}")
