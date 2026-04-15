"""
Generate Phase 2 notebooks: 04_isic2018, 05_coco_amorphous_rigid, 07_cross_dataset_analysis.
Run from repo root: python gen_phase2_notebooks.py
"""
import json
from pathlib import Path

NB_DIR = Path("notebooks")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def code(source: str, cell_id: str) -> dict:
    lines = source.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        pass  # last line has no trailing newline (notebook convention)
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }

def md(source: str, cell_id: str) -> dict:
    lines = source.splitlines(keepends=True)
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": lines,
    }

def nb(cells: list) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "A100"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }

def write_nb(cells, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb(cells), f, ensure_ascii=False, indent=1)
    print(f"Written: {path} ({len(cells)} cells)")


# ─── SHARED INFRASTRUCTURE (shared across notebooks 04, 05) ───────────────────

INSTALL_CELL = '''\
# --- Install pinned dependencies
!pip install --upgrade pip -q
!pip install -U "ultralytics==8.4.9" "wandb==0.24.1" -q
print("Dependencies installed.")'''

CLONE_CELL = '''\
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
print(f"Working directory: {os.getcwd()}")'''

def wandb_cell(project_name: str) -> str:
    return f'''\
# --- WandB setup
import os, wandb
WANDB_PROJECT = "{project_name}"
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
    print(f"WandB logged in. Project: {{WANDB_PROJECT}}")
else:
    print("WandB API key not found — training will proceed without W&B logging.")'''

def drive_cell(drive_exp_path: str) -> str:
    return f'''\
# --- Google Drive: mount + restore completed runs
import shutil

def mount_drive():
    global DRIVE_AVAILABLE
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        DRIVE_EXPERIMENTS.mkdir(parents=True, exist_ok=True)
        DRIVE_AVAILABLE = True
        print(f"Drive mounted. Backup dir: {{DRIVE_EXPERIMENTS}}")
    except Exception as e:
        print(f"Drive not available ({{e}}). Running without Drive persistence.")
        DRIVE_AVAILABLE = False
    return DRIVE_AVAILABLE

def restore_from_drive():
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
        if (drive_run / "results.csv").exists() and not (local_run / "results.csv").exists():
            shutil.copytree(str(drive_run), str(local_run), dirs_exist_ok=True)
            restored += 1
            print(f"  [RESTORE] {{drive_run.name}}")
    msg = "Nothing to restore." if restored == 0 else f"Restored {{restored}} run(s) from Drive."
    print(msg)

mount_drive()
restore_from_drive()'''

NOISE_CELL = '''\
# --- Build noise-perturbed validation splits (idempotent)
# sigma_low=0.02: ~2% of image dim; sigma_high=0.08: ~8% (stress test)
import numpy as np, shutil
from pathlib import Path as _P

SIGMA_LOW  = 0.02
SIGMA_HIGH = 0.08

def _jitter_label_file(src_lbl, dst_lbl, sigma, rng):
    lines = _P(src_lbl).read_text().strip().split("\\n")
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
    _P(dst_lbl).write_text("\\n".join(out) + "\\n")

def build_noise_splits(dataset_root, seed=42):
    src_img = dataset_root / "valid" / "images"
    src_lbl = dataset_root / "valid" / "labels"
    rng = np.random.default_rng(seed)
    for split_name, sigma in [("valid_low", SIGMA_LOW), ("valid_high", SIGMA_HIGH)]:
        dst_img = dataset_root / split_name / "images"
        dst_lbl = dataset_root / split_name / "labels"
        if dst_img.exists() and len(list(dst_img.glob("*"))) > 10:
            print(f"  {split_name}: already exists — skipping.")
            continue
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        img_files = sorted(src_img.glob("*"))
        for img_path in img_files:
            lbl_path = src_lbl / (img_path.stem + ".txt")
            dst_img_path = dst_img / img_path.name
            if not dst_img_path.exists():
                shutil.copy2(str(img_path), str(dst_img_path))
            if lbl_path.exists():
                _jitter_label_file(lbl_path, dst_lbl / lbl_path.name, sigma, rng)
        print(f"  {split_name}: {len(img_files)} images processed (sigma={sigma})")

build_noise_splits(DATASET_ROOT)
print("Noise splits ready.")'''

MONKEY_PATCH_CELL = '''\
# --- Monkey-patch BboxLoss.forward to inject custom IoU loss
# BboxLoss.forward signature in ultralytics 8.4.9:
#   forward(self, pred_dist, pred_bboxes, anchor_points,
#           target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride)
import types
import torch
import torch.nn.functional as F
import ultralytics.utils.loss as loss_mod

_ORIGINAL_BBOX_FORWARD = loss_mod.BboxLoss.forward

def _make_bbox_forward(loss_fn_instance):
    def bbox_loss_forward(
        self, pred_dist, pred_bboxes, anchor_points,
        target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride,
    ):
        weight = target_scores.sum(-1)
        iou_loss = loss_fn_instance(
            pred_bboxes[fg_mask], target_bboxes[fg_mask]
        )
        loss_iou = (iou_loss * weight[fg_mask]).sum() / target_scores_sum
        if self.use_dfl:
            target_ltrb = self.bbox2dist(anchor_points, target_bboxes, self.reg_max - 1)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max), target_ltrb[fg_mask]) * weight[fg_mask]
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)
        return loss_iou, loss_dfl
    return types.MethodType(bbox_loss_forward, None).__func__

def patch_loss(loss_fn):
    loss_mod.BboxLoss.forward = _make_bbox_forward(loss_fn)

def restore_loss():
    loss_mod.BboxLoss.forward = _ORIGINAL_BBOX_FORWARD

print("Monkey-patch infrastructure ready.")'''

def loss_registry_cell(top_lambdas=(0.3, 0.5, 0.8)) -> str:
    lambdas_str = ", ".join(str(r) for r in top_lambdas)
    return f'''\
# --- Loss registry: all 8 baselines + top AEIoU rigidity values
from src.losses import (IoULoss, GIoULoss, DIoULoss, CIoULoss,
                        EIoULoss, ECIoULoss, SIoULoss, WIoULoss, AEIoULoss)

BASELINE_LOSS_REGISTRY = {{
    "iou":   IoULoss(reduction="none"),
    "giou":  GIoULoss(reduction="none"),
    "diou":  DIoULoss(reduction="none"),
    "ciou":  CIoULoss(reduction="none"),
    "eiou":  EIoULoss(reduction="none"),
    "eciou": ECIoULoss(reduction="none"),
    "siou":  SIoULoss(reduction="none"),
    "wiou":  WIoULoss(reduction="none"),
}}

# Phase 2 runs top 3 lambda values from Phase 1 Kvasir results
# (r0p3, r0p5, r0p8 consistently rank highest across noise splits)
TOP_AEIOU_LAMBDAS = [{lambdas_str}]
AEIOU_LOSS_REGISTRY = {{
    f"aeiou_r{{str(r).replace('.','p')}}": AEIoULoss(rigidity=r, reduction="none")
    for r in TOP_AEIOU_LAMBDAS
}}

ALL_LOSS_REGISTRY = {{**BASELINE_LOSS_REGISTRY, **AEIOU_LOSS_REGISTRY}}
ALL_LOSS_KEYS = list(BASELINE_LOSS_REGISTRY.keys()) + list(AEIOU_LOSS_REGISTRY.keys())

print(f"Loss registry: {{len(BASELINE_LOSS_REGISTRY)}} baselines + {{len(AEIOU_LOSS_REGISTRY)}} AEIoU = {{len(ALL_LOSS_REGISTRY)}} total")
for k in ALL_LOSS_KEYS:
    print(f"  {{k}}")'''

def run_training_cell(dataset_prefix: str) -> str:
    return f'''\
# --- run_training: train one model run with full Drive backup + resume support
import json, shutil, time
from pathlib import Path as _P
from datetime import datetime
from ultralytics import YOLO

def _load_manifest():
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {{}}

def write_manifest_entry(run_name, meta):
    manifest = _load_manifest()
    manifest[run_name] = meta
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

def sync_to_drive(run_name):
    if not DRIVE_AVAILABLE:
        return
    local_run = EXPERIMENTS / run_name
    drive_run = DRIVE_EXPERIMENTS / run_name
    try:
        shutil.copytree(str(local_run), str(drive_run), dirs_exist_ok=True)
        print(f"  [DRIVE] Synced {{run_name}}")
    except Exception as e:
        print(f"  [DRIVE] Sync failed for {{run_name}}: {{e}}")

def make_epoch_checkpoint_callback(run_name):
    def _on_epoch_end(trainer):
        if not DRIVE_AVAILABLE:
            return
        last_pt = _P(trainer.save_dir) / "weights" / "last.pt"
        if not last_pt.exists():
            return
        drive_weights = DRIVE_EXPERIMENTS / run_name / "weights"
        drive_weights.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(str(last_pt), str(drive_weights / "last.pt"))
        except Exception as e:
            print(f"  [DRIVE] Epoch checkpoint sync failed: {{e}}")
    return _on_epoch_end

def run_training(loss_name, loss_fn, split_name, yaml_path,
                 seed=42, epochs=None, imgsz=None, device=None):
    epochs = epochs if epochs is not None else EPOCHS
    imgsz  = imgsz  if imgsz  is not None else IMGSZ
    device = device if device is not None else DEVICE

    run_name = f"{dataset_prefix}_yolo26n_{{loss_name}}_{{split_name}}_s{{seed}}_e{{epochs}}"
    run_dir  = EXPERIMENTS / run_name

    if (run_dir / "results.csv").exists():
        print(f"[SKIP] {{run_name}}")
        return run_dir

    drive_last_pt = DRIVE_EXPERIMENTS / run_name / "weights" / "last.pt"
    resuming = DRIVE_AVAILABLE and drive_last_pt.exists()

    if resuming:
        print(f"\\n{{'='*70}}\\n[RESUME] {{run_name}}\\n  Checkpoint on Drive — resuming\\n{{'='*70}}")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(drive_last_pt), str(run_dir / "weights" / "last.pt"))
    else:
        print(f"\\n{{'='*70}}\\n[START] {{run_name}}\\n  loss={{loss_name}}  split={{split_name}}  seed={{seed}}\\n{{'='*70}}")

    meta = {{
        "loss": loss_name, "split": split_name, "seed": seed, "epochs": epochs,
        "rigidity": float(getattr(loss_fn, "rigidity", -1) or -1),
        "run_dir": str(run_dir), "timestamp": datetime.now().isoformat(),
        "status": "running", "resumed": resuming,
    }}
    write_manifest_entry(run_name, meta)

    import os as _os
    _os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    _os.environ["WANDB_NAME"]    = run_name

    t_start = time.time()
    try:
        patch_loss(loss_fn)
        if resuming:
            model = YOLO(str(run_dir / "weights" / "last.pt"))
            model.add_callback("on_train_epoch_end", make_epoch_checkpoint_callback(run_name))
            results = model.train(resume=True)
        else:
            model = YOLO(MODEL_PT)
            model.add_callback("on_train_epoch_end", make_epoch_checkpoint_callback(run_name))
            results = model.train(
                data=str(yaml_path), epochs=epochs, imgsz=imgsz,
                project=str(EXPERIMENTS), name=run_name,
                device=device, seed=seed, exist_ok=True,
            )
        try:
            (run_dir / "run_meta.json").write_text(json.dumps(results.results_dict, indent=2))
        except Exception as e:
            print(f"  [WARN] Could not write run_meta.json: {{e}}")
        meta["status"] = "complete"
        meta["elapsed_sec"] = round(time.time() - t_start, 1)
        try:
            import wandb as _wandb
            if _wandb.run is not None:
                _wandb.finish()
        except Exception:
            pass
    except Exception as e:
        print(f"  [ERROR] {{run_name}} failed: {{e}}")
        meta["status"] = "failed"
        meta["error"] = str(e)
        raise
    finally:
        restore_loss()
        write_manifest_entry(run_name, meta)
        sync_to_drive(run_name)

    print(f"[DONE] {{run_name}}")
    return run_dir

print("run_training() ready.")'''

BASELINE_LOOP_CELL = '''\
# --- Baseline training: all 8 losses × 3 splits × seeds
# Completed runs are skipped automatically via results.csv detection.
for loss_name, loss_fn in BASELINE_LOSS_REGISTRY.items():
    for split_name, cfg_path in SPLIT_CONFIGS.items():
        for seed in SEEDS:
            run_training(loss_name, loss_fn, split_name, cfg_path, seed=seed)

restore_loss()
n = len(BASELINE_LOSS_REGISTRY) * len(SPLIT_CONFIGS) * len(SEEDS)
print(f"\\nBaseline grid complete ({n} runs).")'''

AEIOU_LOOP_CELL = '''\
# --- AEIoU top-lambda training: top 3 lambda values × 3 splits × seeds
for loss_name, loss_fn in AEIOU_LOSS_REGISTRY.items():
    for split_name, cfg_path in SPLIT_CONFIGS.items():
        for seed in SEEDS:
            run_training(loss_name, loss_fn, split_name, cfg_path, seed=seed)

restore_loss()
n = len(AEIOU_LOSS_REGISTRY) * len(SPLIT_CONFIGS) * len(SEEDS)
print(f"\\nAEIoU grid complete ({n} runs).")'''

def results_load_cell(dataset_prefix: str) -> str:
    return f'''\
# --- Load all results.csv files into a flat DataFrame
import pandas as pd

CACHE_CSV = EXPERIMENTS / "all_results_combined.csv"

def load_all_results(force_rebuild=False):
    if CACHE_CSV.exists() and not force_rebuild:
        print(f"Loading from cache: {{CACHE_CSV}}")
        return pd.read_csv(CACHE_CSV)
    print("Building combined results...")
    dfs = []
    for loss_name in ALL_LOSS_KEYS:
        for split_name in SPLIT_CONFIGS:
            for seed in SEEDS:
                run_name = f"{dataset_prefix}_yolo26n_{{loss_name}}_{{split_name}}_s{{seed}}_e{{EPOCHS}}"
                csv_path = EXPERIMENTS / run_name / "results.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    df.columns = df.columns.str.strip()
                    df["run_name"] = run_name
                    df["loss"]     = loss_name
                    df["split"]    = split_name
                    df["seed"]     = seed
                    df["rigidity"] = float(loss_name.split("_r")[1].replace("p",".")) if "aeiou_r" in loss_name else -1.0
                    dfs.append(df)
    if not dfs:
        print("No results found yet.")
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(CACHE_CSV, index=False)
    print(f"Saved {{len(dfs)}} runs to {{CACHE_CSV}}")
    return combined

df_all = load_all_results()
print(f"Total rows: {{len(df_all)}}  |  Runs: {{df_all['run_name'].nunique() if len(df_all) > 0 else 0}}")'''

PIVOT_CELL = '''\
# --- Summary pivot table: final-epoch mAP per loss × split
import numpy as np

MAP95_COL = "metrics/mAP50-95(B)"
MAP50_COL = "metrics/mAP50(B)"

if len(df_all) == 0:
    print("No data yet — run training cells first.")
else:
    df_final = df_all.sort_values("epoch").groupby("run_name").last().reset_index()
    pivot95 = df_final.pivot_table(index="loss", columns="split", values=MAP95_COL, aggfunc="mean")
    for col in ["clean", "low", "high"]:
        if col not in pivot95.columns:
            pivot95[col] = float("nan")
    pivot95 = pivot95[["clean", "low", "high"]]
    pivot95["drop_clean→high"] = pivot95["clean"] - pivot95["high"]
    pivot95 = pivot95.sort_values("clean", ascending=False)
    print("=== Final-epoch mAP50-95 per loss × split ===")
    print(pivot95.round(4).to_string())
    pivot95.to_csv(ANALYSIS_DIR / "pivot_map95.csv")'''

BAR_CHART_CELL = '''\
# --- Bar chart: final mAP50-95 grouped by split
import matplotlib.pyplot as plt
import numpy as np

if len(df_all) == 0:
    print("No data yet.")
else:
    splits = ["clean", "low", "high"]
    split_colors = {"clean": "#4CAF50", "low": "#FFC107", "high": "#F44336"}
    x_labels = ALL_LOSS_KEYS
    x_pos = np.arange(len(x_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 5))
    for i, split in enumerate(splits):
        vals = [pivot95.loc[l, split] if l in pivot95.index else 0 for l in x_labels]
        ax.bar(x_pos + i*width, vals, width, color=split_colors[split], alpha=0.85, label=split)

    ax.axvline(x=len(BASELINE_LOSS_REGISTRY) - 0.5, color="black", linestyle=":", lw=1.5, alpha=0.4)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(x_labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("mAP@[.5:.95]")
    ax.set_title(f"Final mAP50-95 — {DATASET_NAME}", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ANALYSIS_DIR / "fig_map95_bar.png", dpi=150)
    plt.show()
    print("Saved fig_map95_bar.png")'''

FINAL_SYNC_CELL = '''\
# --- Final sync: push all analysis figures and manifest to Drive
import shutil

if DRIVE_AVAILABLE:
    drive_analysis = DRIVE_EXPERIMENTS / "analysis"
    drive_analysis.mkdir(parents=True, exist_ok=True)
    if ANALYSIS_DIR.exists():
        shutil.copytree(str(ANALYSIS_DIR), str(drive_analysis), dirs_exist_ok=True)
        n = len(list(drive_analysis.glob("*.png")))
        print(f"Analysis figures synced: {n} PNGs")
    for fname in ["manifest.json", "all_results_combined.csv"]:
        src = EXPERIMENTS / fname
        if src.exists():
            shutil.copy2(str(src), str(DRIVE_EXPERIMENTS / fname))
            print(f"  Synced {fname}")
    print(f"\\nAll artifacts backed up to: {DRIVE_EXPERIMENTS}")
else:
    print("Drive not mounted — skipping final sync.")'''


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 04: ISIC 2018
# ═══════════════════════════════════════════════════════════════════════════════

ISIC_CONSTANTS = '''\
# --- Constants for ISIC 2018 experiment
import math, time
from pathlib import Path
from datetime import datetime

DATASET_NAME  = "ISIC 2018"
PROJECT_DIR   = Path("/content/amorphous-yolo")
DATASET_ROOT  = PROJECT_DIR / "datasets" / "isic-2018"
EXPERIMENTS   = PROJECT_DIR / "experiments_isic"
ANALYSIS_DIR  = EXPERIMENTS / "analysis"
MANIFEST_PATH = EXPERIMENTS / "manifest.json"

EXPERIMENTS.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

DRIVE_ROOT        = Path("/content/drive/MyDrive/amorphous_yolo")
DRIVE_EXPERIMENTS = DRIVE_ROOT / "experiments_isic"
DRIVE_AVAILABLE   = False

EPOCHS   = 20
IMGSZ    = 640
DEVICE   = 0
MODEL_PT = "yolo26n.pt"
SEEDS    = [42]

SPLIT_CONFIGS = {
    "clean": PROJECT_DIR / "data" / "isic_2018.yaml",
    "low":   PROJECT_DIR / "data" / "isic_2018_low.yaml",
    "high":  PROJECT_DIR / "data" / "isic_2018_high.yaml",
}

print("Constants loaded.")
print(f"  DATASET     : {DATASET_NAME}")
print(f"  EXPERIMENTS : {EXPERIMENTS}")
print(f"  Splits      : {list(SPLIT_CONFIGS.keys())}")'''

ISIC_DOWNLOAD = '''\
# --- Download ISIC 2018 Task 1 (skin lesion segmentation → bounding boxes)
# Total download: ~3 GB (training input + masks + validation input + masks)
# Source: ISIC Challenge Archive (public S3, no auth required)
import zipfile, urllib.request
from pathlib import Path

ISIC_URLS = {
    "ISIC2018_Task1-2_Training_Input.zip":      "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip",
    "ISIC2018_Task1_Training_GroundTruth.zip":  "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip",
    "ISIC2018_Task1-2_Validation_Input.zip":    "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Validation_Input.zip",
    "ISIC2018_Task1_Validation_GroundTruth.zip":"https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Validation_GroundTruth.zip",
}

dl_dir = PROJECT_DIR / "datasets" / "isic-2018-raw"
dl_dir.mkdir(parents=True, exist_ok=True)

# Check if already extracted
train_img_dir = DATASET_ROOT / "train" / "images"
if train_img_dir.exists() and len(list(train_img_dir.glob("*.jpg"))) > 2000:
    print(f"ISIC 2018 already prepared ({len(list(train_img_dir.glob('*.jpg')))} train images) — skipping download.")
else:
    for fname, url in ISIC_URLS.items():
        dest = dl_dir / fname
        if not dest.exists():
            print(f"Downloading {fname} ...")
            urllib.request.urlretrieve(url, dest)
            print(f"  OK ({dest.stat().st_size // 1024 // 1024} MB)")
        else:
            print(f"  {fname} already downloaded.")
        print(f"  Extracting {fname} ...")
        with zipfile.ZipFile(dest, "r") as z:
            z.extractall(dl_dir)
    print("Download and extraction complete.")'''

ISIC_MASK2BBOX = '''\
# --- Convert ISIC 2018 segmentation masks → YOLO bbox labels + train/val split
# ISIC masks are PNG files: white (255) = lesion, black (0) = background.
# We extract the bounding box of all non-zero pixels (same pattern as Kvasir).
# 80/20 train/val split on the 2594 training images. The 100 validation images
# become the held-out test set (not used for training or validation here).
import cv2, numpy as np, random, shutil

TRAIN_DIR = DATASET_ROOT / "train"
VALID_DIR = DATASET_ROOT / "valid"

def _mask_to_yolo_bbox(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    H, W = mask.shape
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    cx = (x + w / 2) / W
    cy = (y + h / 2) / H
    bw = w / W
    bh = h / H
    return cx, cy, bw, bh

if TRAIN_DIR.exists() and len(list((TRAIN_DIR / "images").glob("*.jpg"))) > 2000:
    print(f"Prepared split already exists — skipping mask conversion.")
else:
    raw_img_dir  = PROJECT_DIR / "datasets" / "isic-2018-raw" / "ISIC2018_Task1-2_Training_Input"
    raw_mask_dir = PROJECT_DIR / "datasets" / "isic-2018-raw" / "ISIC2018_Task1_Training_GroundTruth"

    # List all images that have corresponding masks
    all_imgs = sorted(raw_img_dir.glob("ISIC_*.jpg"))
    random.seed(42)
    random.shuffle(all_imgs)
    n_train = int(0.8 * len(all_imgs))
    splits = {"train": all_imgs[:n_train], "valid": all_imgs[n_train:]}

    for split_name, img_list in splits.items():
        img_out = DATASET_ROOT / split_name / "images"
        lbl_out = DATASET_ROOT / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        ok = 0
        for img_path in img_list:
            mask_path = raw_mask_dir / (img_path.stem + "_segmentation.png")
            if not mask_path.exists():
                continue
            bbox = _mask_to_yolo_bbox(mask_path)
            if bbox is None:
                continue
            cx, cy, bw, bh = bbox
            # Copy image
            shutil.copy2(str(img_path), str(img_out / img_path.name))
            # Write YOLO label
            (lbl_out / (img_path.stem + ".txt")).write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\\n")
            ok += 1
        print(f"  {split_name}: {ok} images processed")

    print(f"\\nISIC 2018 prepared: {DATASET_ROOT}")'''

VERIFY_ISIC = '''\
# --- Verify ISIC dataset splits
import yaml

print(f"{'Split':<10} {'YAML':<40} {'Images':<10} {'Labels':<10}")
print("-" * 72)
for split_name, yaml_path in SPLIT_CONFIGS.items():
    assert yaml_path.exists(), f"Missing YAML: {yaml_path}"
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    val_dir = Path(cfg["path"]) / cfg["val"]
    lbl_dir = val_dir.parent.parent / "labels" / val_dir.name.split("/")[-1]
    n_img = len(list(val_dir.glob("*"))) if val_dir.exists() else 0
    n_lbl = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
    print(f"  {split_name:<10} {yaml_path.name:<40} {n_img:<10} {n_lbl:<10}")
print("\\nAll YAML configs verified.")'''

def build_nb04():
    cells = [
        md('# 04 · ISIC 2018: AEIoU vs IoU-Family Baselines\n\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nipun-taneja/amorphous-yolo/blob/main/notebooks/04_isic2018.ipynb)\n\n**Phase 2 — Dataset 1:** Skin lesion detection on ISIC 2018. Validates that AEIoU generalises beyond Kvasir-SEG polyps to a second medical amorphous-object domain.\n\n- **Dataset:** ISIC 2018 Task 1, 2594 training images, 1 class (skin lesion)\n- **Losses:** 8 IoU-family baselines + AEIoU at λ={0.3, 0.5, 0.8}\n- **Protocol:** 3 noise splits (clean / low σ=0.02 / high σ=0.08), seed=42, 20 epochs', "h1"),
        md('## Section 1 · Environment Setup', "s1"),
        code(INSTALL_CELL, "c-install"),
        code(CLONE_CELL, "c-clone"),
        code(ISIC_CONSTANTS, "c-const"),
        md('## Section 2 · ISIC 2018 Dataset\n\nISIC 2018 (International Skin Imaging Collaboration) provides dermoscopic images of skin lesions with pixel-accurate segmentation masks. Skin lesions are **amorphous** — they have diffuse, irregular, biologically non-rigid boundaries that make precise bounding box annotation inherently noisy. This makes ISIC an ideal second test domain for AEIoU.', "s2"),
        code(wandb_cell("amorphous-yolo-isic"), "c-wandb"),
        code(drive_cell("/content/drive/MyDrive/amorphous_yolo/experiments_isic"), "c-drive"),
        code(ISIC_DOWNLOAD, "c-download"),
        code(ISIC_MASK2BBOX, "c-mask2bbox"),
        code(NOISE_CELL, "c-noise"),
        code(VERIFY_ISIC, "c-verify"),
        md('## Section 3 · Monkey-Patch Infrastructure\n\nSame pattern as notebook 03 — see that notebook for full commentary.', "s3"),
        code(MONKEY_PATCH_CELL, "c-patch"),
        md('## Section 4 · Loss Registry\n\nAll 8 standard IoU-family baselines + top 3 AEIoU λ values from Phase 1 Kvasir results (λ=0.3, 0.5, 0.8 were the top performers).', "s4"),
        code(loss_registry_cell((0.3, 0.5, 0.8)), "c-registry"),
        md('## Section 5 · Training Infrastructure', "s5"),
        code(run_training_cell("isic"), "c-run-training"),
        md('## Section 6 · Baseline Training (8 losses × 3 splits)\n\n24 runs total. EIoU is the key baseline for paper comparison; others validate the full IoU family.', "s6"),
        code(BASELINE_LOOP_CELL, "c-baseline-loop"),
        md('## Section 7 · AEIoU Top-Lambda Grid (3 λ × 3 splits)\n\n9 additional runs. λ=0.3, 0.5, 0.8 capture the full range of AEIoU behaviour.', "s7"),
        code(AEIOU_LOOP_CELL, "c-aeiou-loop"),
        md('## Section 8 · Results', "s8"),
        code(results_load_cell("isic"), "c-load"),
        code(PIVOT_CELL, "c-pivot"),
        md('## Section 9 · Analysis', "s9"),
        code(BAR_CHART_CELL, "c-bar"),
        md('## Section 10 · Summary & Drive Sync', "s10"),
        code(FINAL_SYNC_CELL, "c-sync"),
    ]
    return cells


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 05: COCO Amorphous + Rigid
# ═══════════════════════════════════════════════════════════════════════════════

COCO_CONSTANTS = '''\
# --- Constants for COCO amorphous/rigid experiment
import math, time
from pathlib import Path
from datetime import datetime

DATASET_NAME  = "COCO 2017"
PROJECT_DIR   = Path("/content/amorphous-yolo")
EXPERIMENTS_AMORPHOUS = PROJECT_DIR / "experiments_coco_amorphous"
EXPERIMENTS_RIGID     = PROJECT_DIR / "experiments_coco_rigid"
ANALYSIS_DIR_AMORPHOUS = EXPERIMENTS_AMORPHOUS / "analysis"
ANALYSIS_DIR_RIGID     = EXPERIMENTS_RIGID / "analysis"
MANIFEST_PATH_AMORPHOUS = EXPERIMENTS_AMORPHOUS / "manifest.json"
MANIFEST_PATH_RIGID     = EXPERIMENTS_RIGID / "manifest.json"

for d in [EXPERIMENTS_AMORPHOUS, EXPERIMENTS_RIGID,
          ANALYSIS_DIR_AMORPHOUS, ANALYSIS_DIR_RIGID]:
    d.mkdir(parents=True, exist_ok=True)

DRIVE_ROOT               = Path("/content/drive/MyDrive/amorphous_yolo")
DRIVE_EXPERIMENTS_AMORPHOUS = DRIVE_ROOT / "experiments_coco_amorphous"
DRIVE_EXPERIMENTS_RIGID     = DRIVE_ROOT / "experiments_coco_rigid"
DRIVE_AVAILABLE = False

EPOCHS   = 20
IMGSZ    = 640
DEVICE   = 0
MODEL_PT = "yolo26n.pt"
SEEDS    = [42]

# Amorphous COCO categories: deformable objects with ambiguous extents
COCO_AMORPHOUS_CATEGORIES = [
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
    "broccoli", "pizza", "cake", "potted plant", "teddy bear"
]

# Rigid COCO categories: objects with well-defined hard boundaries (control)
COCO_RIGID_CATEGORIES = [
    "car", "bus", "truck", "chair", "dining table",
    "laptop", "tv", "microwave", "refrigerator", "cell phone"
]

SPLIT_CONFIGS_AMORPHOUS = {
    "clean": PROJECT_DIR / "data" / "coco_amorphous.yaml",
    "low":   PROJECT_DIR / "data" / "coco_amorphous_low.yaml",
    "high":  PROJECT_DIR / "data" / "coco_amorphous_high.yaml",
}

SPLIT_CONFIGS_RIGID = {
    "clean": PROJECT_DIR / "data" / "coco_rigid.yaml",
}

print("Constants loaded.")
print(f"  Amorphous categories ({len(COCO_AMORPHOUS_CATEGORIES)}): {COCO_AMORPHOUS_CATEGORIES}")
print(f"  Rigid categories ({len(COCO_RIGID_CATEGORIES)}): {COCO_RIGID_CATEGORIES}")'''

COCO_DOWNLOAD = '''\
# --- Download COCO 2017 subsets using FiftyOne
# fiftyone downloads only the images + annotations for selected categories —
# no need to download the full 25 GB COCO dataset.
!pip install fiftyone -q

import fiftyone as fo
import fiftyone.zoo as foz
import json, shutil, cv2
from pathlib import Path

COCO_AMORPHOUS_ROOT = PROJECT_DIR / "datasets" / "coco-amorphous"
COCO_RIGID_ROOT     = PROJECT_DIR / "datasets" / "coco-rigid"

def download_coco_subset(categories, out_root, split="train", max_samples=5000):
    """Download a COCO subset via FiftyOne and export to YOLO format."""
    out_img_dir = out_root / split / "images"
    out_lbl_dir = out_root / split / "labels"
    if out_img_dir.exists() and len(list(out_img_dir.glob("*.jpg"))) > 100:
        print(f"  {out_root.name}/{split}: already present — skipping.")
        return
    print(f"Downloading COCO {split} subset: {categories[:3]}... ({max_samples} max)")
    dataset = foz.load_zoo_dataset(
        "coco-2017", split=split,
        label_types=["detections"],
        classes=categories,
        max_samples=max_samples,
        dataset_name=f"coco_{out_root.name}_{split}",
        overwrite=True,
    )
    # Build class name → index mapping (sorted for reproducibility)
    sorted_cats = sorted(categories)
    cat2idx = {c: i for i, c in enumerate(sorted_cats)}
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    exported = 0
    for sample in dataset:
        img_src = Path(sample.filepath)
        if not img_src.exists():
            continue
        img = cv2.imread(str(img_src))
        if img is None:
            continue
        H, W = img.shape[:2]
        lines = []
        if sample.ground_truth is not None:
            for det in sample.ground_truth.detections:
                if det.label not in cat2idx:
                    continue
                x, y, w, h = det.bounding_box  # fiftyone: relative [0,1]
                cx = x + w / 2
                cy = y + h / 2
                lines.append(f"{cat2idx[det.label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        if not lines:
            continue
        shutil.copy2(str(img_src), str(out_img_dir / img_src.name))
        (out_lbl_dir / (img_src.stem + ".txt")).write_text("\\n".join(lines) + "\\n")
        exported += 1
    print(f"  Exported {exported} images to {out_img_dir}")

# Download amorphous subset (train + val)
download_coco_subset(COCO_AMORPHOUS_CATEGORIES, COCO_AMORPHOUS_ROOT, split="train", max_samples=8000)
download_coco_subset(COCO_AMORPHOUS_CATEGORIES, COCO_AMORPHOUS_ROOT, split="validation", max_samples=2000)
# Rename fiftyone "validation" → "valid"
for root in [COCO_AMORPHOUS_ROOT]:
    val_dir = root / "validation"
    if val_dir.exists() and not (root / "valid").exists():
        val_dir.rename(root / "valid")

# Download rigid subset (train + val — control group, clean split only)
download_coco_subset(COCO_RIGID_CATEGORIES, COCO_RIGID_ROOT, split="train", max_samples=5000)
download_coco_subset(COCO_RIGID_CATEGORIES, COCO_RIGID_ROOT, split="validation", max_samples=1000)
for root in [COCO_RIGID_ROOT]:
    val_dir = root / "validation"
    if val_dir.exists() and not (root / "valid").exists():
        val_dir.rename(root / "valid")

print("\\nCOCO subsets ready.")'''

COCO_NOISE = '''\
# --- Build noise splits for COCO amorphous (rigid subset uses clean only)
import numpy as np, shutil
from pathlib import Path as _P

SIGMA_LOW  = 0.02
SIGMA_HIGH = 0.08

def _jitter_label_file(src_lbl, dst_lbl, sigma, rng):
    lines = _P(src_lbl).read_text().strip().split("\\n")
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
    _P(dst_lbl).write_text("\\n".join(out) + "\\n")

def build_noise_splits(dataset_root, seed=42):
    src_img = dataset_root / "valid" / "images"
    src_lbl = dataset_root / "valid" / "labels"
    if not src_img.exists():
        print(f"  {dataset_root.name}/valid/images not found — skipping noise splits.")
        return
    rng = np.random.default_rng(seed)
    for split_name, sigma in [("valid_low", SIGMA_LOW), ("valid_high", SIGMA_HIGH)]:
        dst_img = dataset_root / split_name / "images"
        dst_lbl = dataset_root / split_name / "labels"
        if dst_img.exists() and len(list(dst_img.glob("*"))) > 10:
            print(f"  {split_name}: already exists — skipping.")
            continue
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        img_files = sorted(src_img.glob("*"))
        for img_path in img_files:
            lbl_path = src_lbl / (img_path.stem + ".txt")
            if not (dst_img / img_path.name).exists():
                shutil.copy2(str(img_path), str(dst_img / img_path.name))
            if lbl_path.exists():
                _jitter_label_file(lbl_path, dst_lbl / lbl_path.name, sigma, rng)
        print(f"  {split_name}: {len(img_files)} images processed (sigma={sigma})")

build_noise_splits(COCO_AMORPHOUS_ROOT)
print("COCO noise splits ready (rigid subset uses clean only).")'''

COCO_AMORPHOUS_LOOP = '''\
# --- Amorphous COCO: all 8 baselines + top AEIoU × 3 splits
# Set active globals to amorphous experiment dirs before training.
EXPERIMENTS   = EXPERIMENTS_AMORPHOUS
ANALYSIS_DIR  = ANALYSIS_DIR_AMORPHOUS
MANIFEST_PATH = MANIFEST_PATH_AMORPHOUS
DRIVE_EXPERIMENTS = DRIVE_EXPERIMENTS_AMORPHOUS
SPLIT_CONFIGS = SPLIT_CONFIGS_AMORPHOUS
DATASET_NAME  = "COCO Amorphous"

for loss_name, loss_fn in {**BASELINE_LOSS_REGISTRY, **AEIOU_LOSS_REGISTRY}.items():
    for split_name, cfg_path in SPLIT_CONFIGS.items():
        for seed in SEEDS:
            run_training(loss_name, loss_fn, split_name, cfg_path, seed=seed)

restore_loss()
print("\\nCOCO Amorphous training complete.")'''

COCO_RIGID_LOOP = '''\
# --- Rigid COCO: EIoU + CIoU + AEIoU top-lambda × clean split only (control)
# This control group shows AEIoU does not hurt performance on rigid objects.
EXPERIMENTS   = EXPERIMENTS_RIGID
ANALYSIS_DIR  = ANALYSIS_DIR_RIGID
MANIFEST_PATH = MANIFEST_PATH_RIGID
DRIVE_EXPERIMENTS = DRIVE_EXPERIMENTS_RIGID
SPLIT_CONFIGS = SPLIT_CONFIGS_RIGID
DATASET_NAME  = "COCO Rigid (control)"

RIGID_LOSSES = {
    "eiou":       BASELINE_LOSS_REGISTRY["eiou"],
    "ciou":       BASELINE_LOSS_REGISTRY["ciou"],
    "aeiou_r0p3": AEIOU_LOSS_REGISTRY["aeiou_r0p3"],
    "aeiou_r0p5": AEIOU_LOSS_REGISTRY["aeiou_r0p5"],
    "aeiou_r0p8": AEIOU_LOSS_REGISTRY["aeiou_r0p8"],
}
for loss_name, loss_fn in RIGID_LOSSES.items():
    for split_name, cfg_path in SPLIT_CONFIGS.items():
        for seed in SEEDS:
            run_training(loss_name, loss_fn, split_name, cfg_path, seed=seed)

restore_loss()
print("\\nCOCO Rigid control training complete.")'''

COCO_DELTA_PLOT = '''\
# --- Key plot: AEIoU delta vs EIoU on amorphous vs rigid COCO
# The hypothesis: AEIoU gain is LARGER on amorphous objects, near-zero on rigid.
import pandas as pd
import matplotlib.pyplot as plt

def load_pivot(exp_dir, prefix, split="clean"):
    cache = exp_dir / "all_results_combined.csv"
    if not cache.exists():
        return None
    df = pd.read_csv(cache)
    df_last = df.sort_values("epoch").groupby("run_name").last().reset_index()
    pivot = df_last[df_last["split"] == split].pivot_table(
        index="loss", values="metrics/mAP50-95(B)", aggfunc="mean"
    )
    return pivot

pivot_amorphous = load_pivot(EXPERIMENTS_AMORPHOUS, "coco_amorphous")
pivot_rigid     = load_pivot(EXPERIMENTS_RIGID,     "coco_rigid")

if pivot_amorphous is not None and pivot_rigid is not None:
    eiou_a = pivot_amorphous.loc["eiou", "metrics/mAP50-95(B)"] if "eiou" in pivot_amorphous.index else None
    eiou_r = pivot_rigid.loc["eiou", "metrics/mAP50-95(B)"]     if "eiou" in pivot_rigid.index else None

    if eiou_a and eiou_r:
        aeiou_keys = [k for k in pivot_amorphous.index if "aeiou" in k]
        delta_a = {k: pivot_amorphous.loc[k, "metrics/mAP50-95(B)"] - eiou_a for k in aeiou_keys if k in pivot_amorphous.index}
        delta_r = {k: pivot_rigid.loc[k, "metrics/mAP50-95(B)"] - eiou_r for k in aeiou_keys if k in pivot_rigid.index}

        keys = sorted(set(delta_a) & set(delta_r))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter([delta_a[k] for k in keys], [delta_r.get(k, 0) for k in keys], s=80, zorder=5)
        for k in keys:
            ax.annotate(k, (delta_a[k], delta_r.get(k, 0)), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel("Δ mAP50-95 vs EIoU — Amorphous COCO")
        ax.set_ylabel("Δ mAP50-95 vs EIoU — Rigid COCO (control)")
        ax.set_title("AEIoU Delta: Amorphous vs Rigid Objects\\n(ideal: right of 0 for x, near 0 for y)", fontweight="bold")
        fig.tight_layout()
        out = ANALYSIS_DIR_AMORPHOUS / "fig_amorphous_vs_rigid_delta.png"
        fig.savefig(out, dpi=150)
        plt.show()
        print(f"Saved: {out}")
else:
    print("Results not yet available — run training cells first.")'''

def build_nb05():
    cells = [
        md('# 05 · COCO 2017: Amorphous vs Rigid — AEIoU Control Experiment\n\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nipun-taneja/amorphous-yolo/blob/main/notebooks/05_coco_amorphous_rigid.ipynb)\n\n**Phase 2 — Dataset 2:** COCO 2017 split into amorphous and rigid subsets. The key claim: AEIoU gains are larger on deformable/amorphous objects than on rigid ones.\n\n- **Amorphous subset (15 classes):** cat, dog, bird, horse, sheep, cow, elephant, bear, zebra, giraffe, broccoli, pizza, cake, teddy bear, potted plant\n- **Rigid subset (10 classes, control):** car, bus, truck, chair, dining table, laptop, tv, microwave, refrigerator, cell phone\n- **Protocol:** 3 noise splits on amorphous; clean-only on rigid (control)', "h1"),
        md('## Section 1 · Environment Setup', "s1"),
        code(INSTALL_CELL, "c-install"),
        code(CLONE_CELL, "c-clone"),
        code(COCO_CONSTANTS, "c-const"),
        md('## Section 2 · COCO Dataset Setup\n\nDownloads only the selected category subsets via FiftyOne — no need to download the full 25 GB COCO dataset.', "s2"),
        code(wandb_cell("amorphous-yolo-coco"), "c-wandb"),
        code('''\
# --- Google Drive: mount + restore (amorphous subset)
import shutil

def mount_drive():
    global DRIVE_AVAILABLE
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        DRIVE_EXPERIMENTS_AMORPHOUS.mkdir(parents=True, exist_ok=True)
        DRIVE_EXPERIMENTS_RIGID.mkdir(parents=True, exist_ok=True)
        DRIVE_AVAILABLE = True
        print(f"Drive mounted.")
    except Exception as e:
        print(f"Drive not available ({e}).")
        DRIVE_AVAILABLE = False
    return DRIVE_AVAILABLE

def restore_from_drive_dir(local_exp, drive_exp):
    if not DRIVE_AVAILABLE or not drive_exp.exists():
        return 0
    restored = 0
    for drive_run in sorted(drive_exp.iterdir()):
        if not drive_run.is_dir():
            continue
        local_run = local_exp / drive_run.name
        if (drive_run / "results.csv").exists() and not (local_run / "results.csv").exists():
            shutil.copytree(str(drive_run), str(local_run), dirs_exist_ok=True)
            restored += 1
            print(f"  [RESTORE] {drive_run.name}")
    return restored

mount_drive()
r1 = restore_from_drive_dir(EXPERIMENTS_AMORPHOUS, DRIVE_EXPERIMENTS_AMORPHOUS)
r2 = restore_from_drive_dir(EXPERIMENTS_RIGID, DRIVE_EXPERIMENTS_RIGID)
print(f"Restored {r1} amorphous + {r2} rigid runs from Drive.")''', "c-drive"),
        code(COCO_DOWNLOAD, "c-download"),
        code(COCO_NOISE, "c-noise"),
        md('## Section 3 · Monkey-Patch Infrastructure', "s3"),
        code(MONKEY_PATCH_CELL, "c-patch"),
        md('## Section 4 · Loss Registry', "s4"),
        code(loss_registry_cell((0.3, 0.5, 0.8)), "c-registry"),
        md('## Section 5 · Training Infrastructure', "s5"),
        code(run_training_cell("coco_amorphous"), "c-run-amorphous"),
        md('## Section 6 · Amorphous COCO Training (33 runs)\n\nAll 8 baselines + AEIoU λ={0.3, 0.5, 0.8} × 3 noise splits. Expected time: ~54h on A100 across multiple sessions (resume logic handles restarts automatically).', "s6"),
        code(COCO_AMORPHOUS_LOOP, "c-amorphous-loop"),
        md('## Section 7 · Rigid COCO Training (5 runs — control)\n\nEIoU + CIoU + AEIoU top-λ × clean split only. Expected time: ~2h.', "s7"),
        code('''\
# Override run_training for rigid subset (different prefix)
_orig_run_training = run_training

def run_training(loss_name, loss_fn, split_name, yaml_path, seed=42, **kwargs):
    import json, shutil, time
    from pathlib import Path as _P
    from datetime import datetime
    from ultralytics import YOLO
    epochs = kwargs.get("epochs", EPOCHS)
    imgsz  = kwargs.get("imgsz", IMGSZ)
    device = kwargs.get("device", DEVICE)
    run_name = f"coco_rigid_yolo26n_{loss_name}_{split_name}_s{seed}_e{epochs}"
    run_dir  = EXPERIMENTS / run_name
    if (run_dir / "results.csv").exists():
        print(f"[SKIP] {run_name}")
        return run_dir
    print(f"\\n{'='*70}\\n[START] {run_name}\\n{'='*70}")
    meta = {"loss": loss_name, "split": split_name, "seed": seed, "epochs": epochs,
            "rigidity": float(getattr(loss_fn, "rigidity", -1) or -1),
            "status": "running", "timestamp": datetime.now().isoformat()}
    import json as _json
    MANIFEST_PATH.write_text(_json.dumps({**(_json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}), run_name: meta}, indent=2))
    try:
        patch_loss(loss_fn)
        model = YOLO(MODEL_PT)
        results = model.train(data=str(yaml_path), epochs=epochs, imgsz=imgsz,
                              project=str(EXPERIMENTS), name=run_name, device=device,
                              seed=seed, exist_ok=True)
        meta["status"] = "complete"
    except Exception as e:
        meta["status"] = "failed"; raise
    finally:
        restore_loss()
        MANIFEST_PATH.write_text(_json.dumps({**(_json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}), run_name: meta}, indent=2))
        if DRIVE_AVAILABLE:
            try: shutil.copytree(str(run_dir), str(DRIVE_EXPERIMENTS / run_name), dirs_exist_ok=True)
            except: pass
    print(f"[DONE] {run_name}")
    return run_dir''', "c-rigid-fn"),
        code(COCO_RIGID_LOOP, "c-rigid-loop"),
        md('## Section 8 · Results & Key Analysis', "s8"),
        code(COCO_DELTA_PLOT, "c-delta-plot"),
        md('## Section 9 · Final Drive Sync', "s9"),
        code('''\
# --- Final sync for both experiment dirs
import shutil

for exp_dir, drive_dir in [
    (EXPERIMENTS_AMORPHOUS, DRIVE_EXPERIMENTS_AMORPHOUS),
    (EXPERIMENTS_RIGID,     DRIVE_EXPERIMENTS_RIGID),
]:
    if DRIVE_AVAILABLE:
        (drive_dir / "analysis").mkdir(parents=True, exist_ok=True)
        analysis = exp_dir / "analysis"
        if analysis.exists():
            shutil.copytree(str(analysis), str(drive_dir / "analysis"), dirs_exist_ok=True)
        for fname in ["manifest.json", "all_results_combined.csv"]:
            src = exp_dir / fname
            if src.exists():
                shutil.copy2(str(src), str(drive_dir / fname))
        print(f"Synced {exp_dir.name} → Drive")
    else:
        print("Drive not mounted — skipping sync.")''', "c-sync"),
    ]
    return cells


# ═══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 07: Cross-Dataset Analysis
# ═══════════════════════════════════════════════════════════════════════════════

CROSS_DATASET_LOAD = '''\
# --- Load results from all completed experiment directories
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path("/content/amorphous-yolo")

EXPERIMENT_DIRS = {
    "Kvasir-SEG (polyps)":       PROJECT_DIR / "experiments_kvasir",
    "ISIC 2018 (skin lesions)":  PROJECT_DIR / "experiments_isic",
    "COCO Amorphous":            PROJECT_DIR / "experiments_coco_amorphous",
    "COCO Rigid (control)":      PROJECT_DIR / "experiments_coco_rigid",
}

MAP95_COL = "metrics/mAP50-95(B)"
MAP50_COL = "metrics/mAP50(B)"

dfs = {}
for dataset_name, exp_dir in EXPERIMENT_DIRS.items():
    csv = exp_dir / "all_results_combined.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        df["dataset"] = dataset_name
        dfs[dataset_name] = df
        print(f"  {dataset_name}: {df['run_name'].nunique()} runs loaded")
    else:
        print(f"  {dataset_name}: no results yet (run training notebook first)")

df_all = pd.concat(dfs.values(), ignore_index=True) if dfs else pd.DataFrame()
print(f"\\nTotal rows: {len(df_all)} across {len(dfs)} datasets")'''

CROSS_UNIFIED_TABLE = '''\
# --- Unified cross-dataset comparison table (paper Table 2/3)
import pandas as pd
import numpy as np

if len(df_all) == 0:
    print("No data yet.")
else:
    # Final epoch per run
    df_final = df_all.sort_values("epoch").groupby(["run_name","dataset"]).last().reset_index()

    LOSSES_OF_INTEREST = ["eiou", "ciou", "siou", "wiou", "aeiou_r0p3", "aeiou_r0p5", "aeiou_r0p8"]
    rows = []
    for dataset, group in df_final.groupby("dataset"):
        for loss in LOSSES_OF_INTEREST:
            sub = group[(group["loss"] == loss) & (group["split"] == "clean")]
            if sub.empty:
                continue
            val = sub[MAP95_COL].mean()
            rows.append({"dataset": dataset, "loss": loss, "mAP50-95": val})

    table = pd.DataFrame(rows).pivot_table(
        index="loss", columns="dataset", values="mAP50-95", aggfunc="mean"
    ).round(4)

    print("=== Cross-Dataset mAP50-95 (clean split) ===")
    print(table.to_string())

    ANALYSIS_DIR = PROJECT_DIR / "experiments_kvasir" / "analysis"
    ANALYSIS_DIR.mkdir(exist_ok=True)
    table.to_csv(ANALYSIS_DIR / "cross_dataset_table.csv")
    print(f"\\nSaved cross_dataset_table.csv")'''

CROSS_LAMBDA_PLOT = '''\
# --- Cross-dataset lambda consistency plot
# Does the optimal lambda (~0.3-0.5) hold across all amorphous datasets?
import matplotlib.pyplot as plt
import pandas as pd

if len(df_all) == 0:
    print("No data yet.")
else:
    df_final = df_all.sort_values("epoch").groupby(["run_name","dataset"]).last().reset_index()
    aeiou_df = df_final[df_final["loss"].str.startswith("aeiou_r") & (df_final["split"] == "clean")].copy()
    aeiou_df["lambda"] = aeiou_df["loss"].str.extract(r"aeiou_r([\d]+p[\d]+)")[0].str.replace("p",".")
    aeiou_df["lambda"] = pd.to_numeric(aeiou_df["lambda"], errors="coerce")

    datasets_with_data = aeiou_df["dataset"].unique()
    colors = ["#E63946", "#2196F3", "#4CAF50", "#FF9800"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, dataset in enumerate(datasets_with_data):
        sub = aeiou_df[aeiou_df["dataset"] == dataset].sort_values("lambda")
        if sub.empty:
            continue
        ax.plot(sub["lambda"], sub[MAP95_COL], marker="o", label=dataset,
                color=colors[i % len(colors)], linewidth=2)

    ax.set_xlabel("AEIoU λ (rigidity)", fontsize=11)
    ax.set_ylabel("mAP@[.5:.95] (clean split)", fontsize=11)
    ax.set_title("Cross-Dataset Lambda Consistency\\n(Does optimal λ generalise?)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = PROJECT_DIR / "experiments_kvasir" / "analysis" / "cross_dataset_lambda_curve.png"
    fig.savefig(out, dpi=150)
    plt.show()
    print(f"Saved: {out}")'''

CROSS_DELTA_SCATTER = '''\
# --- Delta scatter: AEIoU improvement over EIoU baseline per dataset
# x-axis: mAP improvement of best AEIoU over EIoU baseline
# Expected: large positive delta on medical/amorphous, near-zero on rigid COCO
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if len(df_all) == 0:
    print("No data yet.")
else:
    df_final = df_all.sort_values("epoch").groupby(["run_name","dataset"]).last().reset_index()
    results = []
    for dataset, group in df_final.groupby("dataset"):
        eiou_row = group[(group["loss"] == "eiou") & (group["split"] == "clean")]
        if eiou_row.empty:
            continue
        eiou_map = eiou_row[MAP95_COL].mean()
        for lam in ["aeiou_r0p3", "aeiou_r0p5", "aeiou_r0p8"]:
            aeiou_row = group[(group["loss"] == lam) & (group["split"] == "clean")]
            if aeiou_row.empty:
                continue
            aeiou_map = aeiou_row[MAP95_COL].mean()
            delta = aeiou_map - eiou_map
            results.append({"dataset": dataset, "lambda": lam, "delta_vs_eiou": delta,
                            "is_rigid": "Rigid" in dataset})

    df_res = pd.DataFrame(results)
    if len(df_res) == 0:
        print("Not enough data for delta scatter.")
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        for dataset, grp in df_res.groupby("dataset"):
            color = "#F44336" if "Rigid" in dataset else "#4CAF50"
            marker = "s" if "Rigid" in dataset else "o"
            ax.scatter(range(len(grp)), grp["delta_vs_eiou"], label=dataset,
                      color=color, marker=marker, s=80, zorder=5)

        ax.axhline(0, color="gray", lw=1.5, linestyle="--", alpha=0.7)
        ax.set_ylabel("Δ mAP50-95 vs EIoU baseline")
        ax.set_title("AEIoU Improvement Over EIoU: Amorphous vs Rigid\\n(positive = AEIoU wins)", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = PROJECT_DIR / "experiments_kvasir" / "analysis" / "cross_dataset_delta_scatter.png"
        fig.savefig(out, dpi=150)
        plt.show()
        print(f"Saved: {out}")'''

NOISE_DEGRADATION_PLOT = '''\
# --- Noise degradation comparison across datasets
# Shows mAP_high / mAP_clean ratio for EIoU vs best AEIoU per dataset
# AEIoU should degrade LESS on high noise (robust to annotation noise)
import matplotlib.pyplot as plt
import pandas as pd

if len(df_all) == 0:
    print("No data yet.")
else:
    df_final = df_all.sort_values("epoch").groupby(["run_name","dataset"]).last().reset_index()
    datasets = [d for d in df_all["dataset"].unique() if "Rigid" not in d]
    losses_to_compare = ["eiou", "aeiou_r0p3", "aeiou_r0p5"]

    rows = []
    for dataset in datasets:
        grp = df_final[df_final["dataset"] == dataset]
        for loss in losses_to_compare:
            clean = grp[(grp["loss"] == loss) & (grp["split"] == "clean")][MAP95_COL].mean()
            high  = grp[(grp["loss"] == loss) & (grp["split"] == "high")][MAP95_COL].mean()
            if pd.notna(clean) and pd.notna(high) and clean > 0:
                rows.append({"dataset": dataset, "loss": loss,
                             "clean": clean, "high": high,
                             "robustness_ratio": high / clean})

    if rows:
        df_rob = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(datasets))
        width = 0.25
        for i, loss in enumerate(losses_to_compare):
            sub = df_rob[df_rob["loss"] == loss]
            vals = [sub[sub["dataset"] == d]["robustness_ratio"].values[0]
                    if len(sub[sub["dataset"] == d]) > 0 else 0 for d in datasets]
            ax.bar(x + i*width, vals, width, label=loss, alpha=0.85)
        ax.axhline(1.0, color="gray", linestyle="--", lw=1)
        ax.set_xticks(x + width)
        ax.set_xticklabels(datasets, rotation=15, ha="right")
        ax.set_ylabel("mAP_high / mAP_clean (higher = more robust)")
        ax.set_title("Noise Robustness: EIoU vs AEIoU Across Datasets", fontweight="bold")
        ax.legend()
        fig.tight_layout()
        out = PROJECT_DIR / "experiments_kvasir" / "analysis" / "cross_dataset_noise_robustness.png"
        fig.savefig(out, dpi=150)
        plt.show()
        print(f"Saved: {out}")
    else:
        print("Not enough data — need results from multiple splits.")'''

def build_nb07():
    cells = [
        md('# 07 · Cross-Dataset Analysis — AEIoU Journal Paper Figures\n\n[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nipun-taneja/amorphous-yolo/blob/main/notebooks/07_cross_dataset_analysis.ipynb)\n\n**Phase 2 Analysis Notebook.** No training — loads CSVs from all experiment directories and generates publication-ready figures.\n\nRun this notebook after completing training in notebooks 03, 04, and 05.', "h1"),
        md('## Section 1 · Environment Setup', "s1"),
        code(INSTALL_CELL, "c-install"),
        code(CLONE_CELL, "c-clone"),
        code('''\
# --- Mount Google Drive and restore experiment results
import os
from pathlib import Path

PROJECT_DIR = Path("/content/amorphous-yolo")
DRIVE_ROOT  = Path("/content/drive/MyDrive/amorphous_yolo")

try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    DRIVE_AVAILABLE = True
    print("Drive mounted.")
except Exception as e:
    DRIVE_AVAILABLE = False
    print(f"Drive not available ({e}). Using local results if present.")

# Restore all experiment directories from Drive to local
import shutil

EXP_DIRS = {
    "experiments_kvasir":        PROJECT_DIR / "experiments_kvasir",
    "experiments_isic":          PROJECT_DIR / "experiments_isic",
    "experiments_coco_amorphous":PROJECT_DIR / "experiments_coco_amorphous",
    "experiments_coco_rigid":    PROJECT_DIR / "experiments_coco_rigid",
}

if DRIVE_AVAILABLE:
    for name, local_dir in EXP_DIRS.items():
        drive_dir = DRIVE_ROOT / name
        if not drive_dir.exists():
            print(f"  {name}: not on Drive yet.")
            continue
        local_dir.mkdir(parents=True, exist_ok=True)
        # Restore all_results_combined.csv and manifest.json
        for fname in ["all_results_combined.csv", "manifest.json"]:
            src = drive_dir / fname
            dst = local_dir / fname
            if src.exists() and not dst.exists():
                shutil.copy2(str(src), str(dst))
                print(f"  [RESTORE] {name}/{fname}")
        # Restore individual run results.csv
        for drive_run in sorted(drive_dir.iterdir()):
            if not drive_run.is_dir():
                continue
            local_run = local_dir / drive_run.name
            if (drive_run / "results.csv").exists() and not (local_run / "results.csv").exists():
                local_run.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(drive_run / "results.csv"), str(local_run / "results.csv"))
    print("\\nDrive restore complete.")''', "c-drive"),
        md('## Section 2 · Load All Results', "s2"),
        code(CROSS_DATASET_LOAD, "c-load"),
        md('## Section 3 · Unified Comparison Table (Paper Table 2/3)', "s3"),
        code(CROSS_UNIFIED_TABLE, "c-table"),
        md('## Section 4 · Cross-Dataset Lambda Consistency\n\n**Key question:** Does the optimal λ≈0.3–0.5 from Kvasir generalise to other amorphous datasets? If yes, this validates AEIoU\'s λ as a dataset-agnostic hyperparameter.', "s4"),
        code(CROSS_LAMBDA_PLOT, "c-lambda"),
        md('## Section 5 · AEIoU Delta vs EIoU Across Datasets\n\n**Key evidence for the paper:** AEIoU improvement is larger on amorphous objects than on rigid ones.', "s5"),
        code(CROSS_DELTA_SCATTER, "c-delta"),
        md('## Section 6 · Noise Robustness Across Datasets\n\nShows mAP_high/mAP_clean ratio — higher means more robust to annotation noise.', "s6"),
        code(NOISE_DEGRADATION_PLOT, "c-noise-robust"),
        md('## Section 7 · Summary for Paper\n\nAll figures saved to `experiments_kvasir/analysis/`. Collect them from Drive for the paper manuscript.', "s7"),
        code('''\
# --- Print summary of available figures
from pathlib import Path

analysis_dir = Path("/content/amorphous-yolo/experiments_kvasir/analysis")
if analysis_dir.exists():
    figs = sorted(analysis_dir.glob("cross_dataset_*.png"))
    print(f"Cross-dataset figures ({len(figs)}):")
    for f in figs:
        print(f"  {f.name}")
else:
    print("Analysis directory not found.")

# Copy all cross-dataset figures to Drive
if DRIVE_AVAILABLE:
    drive_analysis = DRIVE_ROOT / "cross_dataset_analysis"
    drive_analysis.mkdir(parents=True, exist_ok=True)
    import shutil
    for f in analysis_dir.glob("cross_dataset_*.png"):
        shutil.copy2(str(f), str(drive_analysis / f.name))
    print(f"\\nFigures backed up to Drive: {drive_analysis}")''', "c-summary"),
    ]
    return cells


# ─── Write notebooks ──────────────────────────────────────────────────────────
write_nb(build_nb04(), NB_DIR / "04_isic2018.ipynb")
write_nb(build_nb05(), NB_DIR / "05_coco_amorphous_rigid.ipynb")
write_nb(build_nb07(), NB_DIR / "07_cross_dataset_analysis.ipynb")

print("\nAll Phase 2 notebooks generated.")
