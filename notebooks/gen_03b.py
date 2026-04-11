#!/usr/bin/env python3
"""Part 2 of the generator — appends cells 16-49 to the cells list built in gen_03.py.
This file is imported by gen_03_run.py which combines both parts and writes the notebook.
"""

def add_cells_part2(cells):
    from pathlib import Path
    def md(source):
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    def code(source):
        return {"cell_type": "code", "execution_count": None,
                "metadata": {}, "outputs": [], "source": source}

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4 — Monkey-Patch Infrastructure
    # ─────────────────────────────────────────────────────────────────────────
    cells.append(md(
"""## Section 4 · Monkey-Patch Infrastructure

### Why monkey-patch instead of editing Ultralytics source?

Ultralytics is a versioned pip package. Editing its source would:
1. Break on `pip install --upgrade ultralytics`
2. Make the experiment hard to reproduce on a fresh Colab runtime
3. Require maintaining a fork

Instead, we replace `BboxLoss.forward` at runtime with a closure that:
- Calls our custom loss function for the **IoU term**
- Copies the **DFL term** verbatim from Ultralytics 8.4.9
- Is fully reversible via `restore_loss()` — critical between runs

The patch is applied immediately before `model.train()` and removed in a
`try/finally` block so it is guaranteed to be restored even if training crashes.

**What to look for in Cell 18:** Running `patch_loss(EIoULoss(...))` then
`restore_loss()` should leave `BboxLoss.forward` identical to its original form
(same function object reference). The print at the end confirms this.
"""
    ))

    cells.append(code(
"""# --- Full monkey-patch implementation (verbatim pattern from notebook 02)
# BboxLoss.forward signature in ultralytics 8.4.9:
#   forward(self, pred_dist, pred_bboxes, anchor_points,
#           target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride)
import types
import torch
import torch.nn.functional as F
import ultralytics.utils.loss as loss_mod

# Save the original forward method ONCE at import time.
# This is the reference we restore after each training run.
_ORIGINAL_BBOX_FORWARD = loss_mod.BboxLoss.forward


def _make_bbox_forward(loss_fn_instance):
    \"\"\"
    Factory: returns a bound-method replacement for BboxLoss.forward that
    uses loss_fn_instance for the IoU term and keeps DFL unchanged.

    loss_fn_instance must support reduction='none' and return shape [N].
    \"\"\"
    def bbox_loss_forward(
        self,
        pred_dist, pred_bboxes, anchor_points,
        target_bboxes, target_scores, target_scores_sum,
        fg_mask, imgsz, stride,
    ):
        # ── IoU term ──────────────────────────────────────────────────────────
        # target_scores: [B, A, C] — score per anchor; sum over classes gives weight
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)  # shape: [N, 1]

        # Apply our custom loss to foreground anchors only (fg_mask selects them)
        # pred_bboxes and target_bboxes are in xyxy format, normalised to image size
        per_box = loss_fn_instance(
            pred_bboxes[fg_mask],    # [N, 4] xyxy predicted
            target_bboxes[fg_mask],  # [N, 4] xyxy ground truth
        )  # → [N] per-box loss values (reduction='none')

        # Weighted average over foreground anchors, normalised by total score sum
        loss_iou = (per_box.unsqueeze(-1) * weight).sum() / target_scores_sum

        # ── DFL term (copied verbatim from ultralytics 8.4.9) ─────────────────
        # DFL (Distribution Focal Loss) operates on the predicted distance
        # distribution; it is independent of the IoU metric choice.
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
    \"\"\"Apply custom loss patch. Always call restore_loss() before patching again.\"\"\"
    loss_mod.BboxLoss.forward = _make_bbox_forward(loss_fn_instance)
    print(f"  [PATCH] BboxLoss.forward → {type(loss_fn_instance).__name__}"
          + (f"(λ={loss_fn_instance.rigidity})" if hasattr(loss_fn_instance, "rigidity") else ""))


def restore_loss():
    \"\"\"Restore the original Ultralytics CIoU-based BboxLoss.forward.\"\"\"
    loss_mod.BboxLoss.forward = _ORIGINAL_BBOX_FORWARD


print("Patch infrastructure ready.")
print(f"  Original forward saved: {_ORIGINAL_BBOX_FORWARD}")
"""
    ))

    cells.append(code(
"""# --- Verify patch round-trip
from src.losses import EIoULoss

# Step 1: apply patch
patch_loss(EIoULoss(reduction="none"))
patched_forward = loss_mod.BboxLoss.forward
print(f"After patch:   BboxLoss.forward is <function bbox_loss_forward>: "
      f"{'bbox_loss_forward' in str(patched_forward)}")

# Step 2: restore
restore_loss()
restored_forward = loss_mod.BboxLoss.forward
print(f"After restore: BboxLoss.forward is original: "
      f"{restored_forward is _ORIGINAL_BBOX_FORWARD}")

assert restored_forward is _ORIGINAL_BBOX_FORWARD, "restore_loss() failed!"
print("Patch round-trip verified.")
"""
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5 — Loss Registry
    # ─────────────────────────────────────────────────────────────────────────
    cells.append(md(
"""## Section 5 · Loss Registry

All **6 standard IoU-family losses** and **10 AEIoU variants** are instantiated here.
Each loss is monkey-patched into Ultralytics one at a time during training.
Comparing against all baselines (not just EIoU) ensures the paper's claims are
robust: AEIoU must beat the **entire field**, not just one competitor.

| Baseline | Publication | Key innovation |
|---|---|---|
| IoU | Yu et al., 2016 | Pure overlap, no auxiliary penalties |
| GIoU | Rezatofighi et al., 2019 | Enclosing area fill penalty |
| DIoU | Zheng et al., 2020 | Center-distance penalty |
| CIoU | Zheng et al., 2020 | + aspect-ratio consistency penalty |
| EIoU | Zheng et al., 2021 | Decoupled width/height penalty, enclosing-box normalised |
| ECIoU | Zhang et al., 2023 | max(pred,target) normalised |
| **AEIoU** | **Ours** | Target-normalised + λ-rigidity |
"""
    ))

    cells.append(code(
"""# --- Loss registry: all baselines + AEIoU grid
from src.losses import IoULoss, GIoULoss, DIoULoss, CIoULoss, EIoULoss, ECIoULoss, AEIoULoss

# All 6 standard baselines — reduction='none' required by the monkey-patch
BASELINE_LOSS_REGISTRY = {
    "iou":   IoULoss(reduction="none"),
    "giou":  GIoULoss(reduction="none"),
    "diou":  DIoULoss(reduction="none"),
    "ciou":  CIoULoss(reduction="none"),
    "eiou":  EIoULoss(reduction="none"),
    "eciou": ECIoULoss(reduction="none"),
}

# AEIoU grid: 10 instances, one per λ value
AEIOU_LOSS_REGISTRY = {
    f"aeiou_r{_fmt_r(r)}": AEIoULoss(rigidity=r, reduction="none")
    for r in AEIOU_RIGIDITIES
}

# Combined registry for easy iteration
ALL_LOSS_REGISTRY = {**BASELINE_LOSS_REGISTRY, **AEIOU_LOSS_REGISTRY}

print(f"Baselines ({len(BASELINE_LOSS_REGISTRY)}):")
for name, fn in BASELINE_LOSS_REGISTRY.items():
    print(f"  {name}: {type(fn).__name__}")

print(f"\\nAEIoU grid ({len(AEIOU_LOSS_REGISTRY)}):")
for name, fn in AEIOU_LOSS_REGISTRY.items():
    print(f"  {name}: rigidity={fn.rigidity}")

n_total = len(ALL_LOSS_REGISTRY) * len(SPLIT_CONFIGS) * len(SEEDS)
print(f"\\nTotal: {len(ALL_LOSS_REGISTRY)} losses x {len(SPLIT_CONFIGS)} splits "
      f"x {len(SEEDS)} seed(s) = {n_total} runs")
"""
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 6 — Training Loop
    # ─────────────────────────────────────────────────────────────────────────
    cells.append(md(
"""## Section 6 · Training Infrastructure

### Run naming convention
```
kvasir_yolo26n_{loss_key}_{split}_e{epochs}
```
Examples:
- `kvasir_yolo26n_eiou_clean_e20`
- `kvasir_yolo26n_aeiou_r0p3_low_e20`
- `kvasir_yolo26n_aeiou_r1p0_high_e20`

`loss_key` always comes from the registry — never constructed ad-hoc.

### Idempotency
`run_training()` checks if `{run_dir}/results.csv` already exists.
If yes → `[SKIP]`. This means any interrupted session can be resumed by
re-running the training cells without duplicating completed runs.

### Manifest
`experiments_kvasir/manifest.json` is updated after **every** run (even failures).
Status field: `"running"` → `"complete"` or `"failed"`. This allows resuming
interrupted sessions and auditing which runs succeeded.

### GPU requirements
- T4 (16 GB): ~3–5 min/run × 33 runs = ~2.5 hrs total
- A100 (40 GB): ~1–2 min/run = ~1 hr total
- Recommended: rent an A100 or L4 for this notebook

**What to look for:** Each run's stdout should show `[START]` then Ultralytics'
training progress bar. `[SKIP]` means the run was already completed — safe to ignore.
"""
    ))

    cells.append(code(
"""# --- Training functions: run_training + write_manifest_entry + sync_to_drive
# --- + make_epoch_checkpoint_callback (epoch-level Drive sync for mid-run resume)
import json, shutil
from pathlib import Path as _Path
from datetime import datetime
from ultralytics import YOLO

def _load_manifest():
    # Load manifest.json as a dict, or return empty dict if not present.
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def write_manifest_entry(run_name, meta):
    # Atomically update manifest.json with the entry for run_name.
    manifest = _load_manifest()
    manifest[run_name] = meta
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def sync_to_drive(run_name):
    # Copy a single completed run directory from local storage to Google Drive.
    # Called in run_training()'s finally block so every run is persisted
    # immediately — a Colab timeout after this point loses nothing.
    if not DRIVE_AVAILABLE:
        return
    local_run = EXPERIMENTS / run_name
    drive_run = DRIVE_EXPERIMENTS / run_name
    try:
        shutil.copytree(str(local_run), str(drive_run), dirs_exist_ok=True)
        print(f"  [DRIVE] Synced {run_name}")
    except Exception as e:
        print(f"  [DRIVE] Sync failed for {run_name}: {e}")


def make_epoch_checkpoint_callback(run_name):
    # Returns an Ultralytics callback that copies last.pt to Drive after every epoch.
    # Ultralytics calls on_train_epoch_end(trainer) automatically each epoch.
    # trainer.save_dir points to the run directory (e.g. experiments_kvasir/run_name/).
    # yolo26n last.pt is ~20 MB — negligible Drive quota per epoch.
    # This enables mid-run resume: if Colab disconnects at epoch N, the next session
    # restores last.pt from Drive and calls model.train(resume=True) to continue.
    def _on_epoch_end(trainer):
        if not DRIVE_AVAILABLE:
            return
        last_pt = _Path(trainer.save_dir) / "weights" / "last.pt"
        if not last_pt.exists():
            return
        drive_weights = DRIVE_EXPERIMENTS / run_name / "weights"
        drive_weights.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(str(last_pt), str(drive_weights / "last.pt"))
        except Exception as e:
            # Non-fatal: training continues even if the epoch sync fails.
            print(f"  [DRIVE] Epoch checkpoint sync failed at epoch: {e}")
    return _on_epoch_end


def run_training(loss_name, loss_fn, split_name, yaml_path,
                 seed=42, epochs=None, imgsz=None, device=None):
    # Train one YOLO26n model. Returns the experiment run directory Path.
    # Supports three scenarios transparently:
    #   1. Fresh start  — no local or Drive checkpoint
    #   2. Resume       — Drive has last.pt from an interrupted run (no local results.csv)
    #   3. Skip         — local results.csv exists (run already completed)
    epochs = epochs if epochs is not None else EPOCHS
    imgsz  = imgsz  if imgsz  is not None else IMGSZ
    device = device if device is not None else DEVICE

    # Canonical run name — includes seed for multi-seed experiments
    run_name = f"kvasir_yolo26n_{loss_name}_{split_name}_s{seed}_e{epochs}"
    run_dir  = EXPERIMENTS / run_name

    # ── Skip if already completed ──────────────────────────────────────────────
    if (run_dir / "results.csv").exists():
        print(f"[SKIP] {run_name}")
        return run_dir

    # ── Check for a Drive checkpoint from an interrupted run ───────────────────
    # last.pt on Drive means a previous session started this run but did not finish.
    # We restore the checkpoint locally so Ultralytics can resume training.
    drive_last_pt = DRIVE_EXPERIMENTS / run_name / "weights" / "last.pt"
    resuming = DRIVE_AVAILABLE and drive_last_pt.exists()

    if resuming:
        print(f"\\n{'='*70}")
        print(f"[RESUME] {run_name}")
        print(f"  Checkpoint found on Drive — resuming from last saved epoch")
        print(f"{'='*70}")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(drive_last_pt), str(run_dir / "weights" / "last.pt"))
    else:
        print(f"\\n{'='*70}")
        print(f"[START] {run_name}")
        print(f"  loss={loss_name}  split={split_name}  seed={seed}  epochs={epochs}")
        print(f"{'='*70}")

    # Record intent in manifest before training (status='running')
    # This allows detecting crashed runs that never reached status='complete'
    meta = {
        "loss":        loss_name,
        "split":       split_name,
        "seed":        seed,
        "epochs":      epochs,
        "rigidity":    float(getattr(loss_fn, "rigidity", -1) or -1),
        "run_dir":     str(run_dir),
        "weights":     str(run_dir / "weights" / "best.pt"),
        "results_csv": str(run_dir / "results.csv"),
        "timestamp":   datetime.now().isoformat(),
        "status":      "running",
        "resumed":     resuming,
    }
    write_manifest_entry(run_name, meta)

    t_start = time.time()
    try:
        # Name this run in WandB before training starts so it appears correctly
        # in the dashboard even if the session is interrupted mid-run.
        import os as _os
        _os.environ["WANDB_PROJECT"] = WANDB_PROJECT
        _os.environ["WANDB_NAME"]    = run_name
        _os.environ["WANDB_TAGS"]    = f"{loss_name},{split_name}"

        # Apply our custom loss to BboxLoss.forward
        patch_loss(loss_fn)

        if resuming:
            model = YOLO(str(run_dir / "weights" / "last.pt"))
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

        # Snapshot Ultralytics metadata for offline analysis
        try:
            (run_dir / "run_meta.json").write_text(
                json.dumps(results.results_dict, indent=2)
            )
        except Exception as e:
            print(f"  [WARN] Could not write run_meta.json: {e}")

        meta["status"] = "complete"
        meta["elapsed_sec"] = round(time.time() - t_start, 1)

        # Explicitly finish the WandB run so metrics are fully uploaded
        # before we move to the next training run.
        try:
            import wandb as _wandb
            if _wandb.run is not None:
                _wandb.finish()
        except Exception:
            pass

    except Exception as e:
        print(f"  [ERROR] {run_name} failed: {e}")
        meta["status"] = "failed"
        meta["error"]  = str(e)
        raise

    finally:
        # CRITICAL: always restore — never leave Ultralytics in patched state
        restore_loss()
        write_manifest_entry(run_name, meta)
        # Sync this run to Drive immediately — protects against future timeouts.
        # A Colab crash after this line loses at most the next run, never this one.
        sync_to_drive(run_name)

    print(f"[DONE] {run_name}")
    return run_dir


print("run_training() ready.")
print(f"Manifest path: {MANIFEST_PATH}")
"""
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 7 — EIoU Baseline (3 runs)
    # ─────────────────────────────────────────────────────────────────────────
    cells.append(md(
"""## Section 7 · All Baseline Training (IoU, GIoU, DIoU, CIoU, EIoU, ECIoU)

Training all 6 standard IoU-family losses as baselines. Each baseline is trained
on all 3 splits (clean, low-noise, high-noise) and all seeds.

| Loss | Innovation | Splits | Seeds | Runs |
|---|---|---|---|---|
| IoU | Pure overlap | 3 | per SEEDS | 3×N |
| GIoU | + enclosing area | 3 | per SEEDS | 3×N |
| DIoU | + center distance | 3 | per SEEDS | 3×N |
| CIoU | + aspect ratio | 3 | per SEEDS | 3×N |
| EIoU | + decoupled W/H (enclosing-normalised) | 3 | per SEEDS | 3×N |
| ECIoU | + decoupled W/H (max-normalised) | 3 | per SEEDS | 3×N |

**Expected runtime:** ~30-50 min for 18 runs on T4 (1 seed), ~90-150 min with 3 seeds.
"""
    ))

    cells.append(code(
"""# --- Baseline training: all 6 standard losses × 3 splits × N seeds
# Re-running is safe — completed runs are skipped via [SKIP] logic.
for loss_name, loss_fn in BASELINE_LOSS_REGISTRY.items():
    for split_name, cfg_path in SPLIT_CONFIGS.items():
        for seed in SEEDS:
            run_training(
                loss_name=loss_name,
                loss_fn=loss_fn,
                split_name=split_name,
                yaml_path=cfg_path,
                seed=seed,
            )

restore_loss()  # Defensive restore
n_base = len(BASELINE_LOSS_REGISTRY) * len(SPLIT_CONFIGS) * len(SEEDS)
print(f"\\nAll {n_base} baseline runs complete (or skipped).")
"""
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 8 — AEIoU Grid (30 runs)
    # ─────────────────────────────────────────────────────────────────────────
    cells.append(md(
"""## Section 8 · AEIoU Rigidity Grid Search (30 runs)

We sweep λ from 0.1 to 1.0 in steps of 0.1 across all three splits:
- 10 λ values × 3 splits = **30 training runs**

### λ interpretation for polyp detection

| λ | Interpretation | Expected behaviour |
|---|---|---|
| 0.1 | Shape labels nearly ignored — pure center+overlap loss | High robustness, possibly lower clean mAP |
| 0.2–0.4 | Moderate shape trust — expected optimal range | Best clean mAP, good robustness |
| 0.5–0.7 | Stronger shape penalty | Converges to EIoU-like behaviour |
| 1.0 | Full shape penalty (different normaliser than EIoU) | ~= EIoU, confirms normaliser effect is small |

**Note:** `aeiou_r1p0` and `eiou` differ only in the size-term normaliser
(target dims vs enclosing dims). If their mAP is nearly identical, the normaliser
choice has minimal practical impact. If `aeiou_r1p0` noticeably underperforms `eiou`
on clean data, the target-dim normaliser may be too aggressive.

**Expected runtime:** ~90–150 minutes for all 30 runs on T4.
Grab a coffee. Or run overnight with a Colab Pro session.
"""
    ))

    cells.append(code(
"""# --- AEIoU rigidity grid training (10 lambdas x 3 splits x N seeds)
# Outer loop: rigidity values (0.1 -> 1.0)
# Middle loop: splits (clean, low, high)
# Inner loop: seeds
# All runs are idempotent — re-run safely after interruption.

total = len(AEIOU_RIGIDITIES) * len(SPLIT_CONFIGS) * len(SEEDS)
done  = 0

for r in AEIOU_RIGIDITIES:
    loss_name = f"aeiou_r{_fmt_r(r)}"
    loss_fn   = AEIOU_LOSS_REGISTRY[loss_name]

    for split_name, cfg_path in SPLIT_CONFIGS.items():
        for seed in SEEDS:
            done += 1
            print(f"\\n[{done}/{total}] lam={r}  split={split_name}  seed={seed}")
            run_training(
                loss_name=loss_name,
                loss_fn=loss_fn,
                split_name=split_name,
                yaml_path=cfg_path,
                seed=seed,
            )

restore_loss()  # Defensive final restore
print(f"\\nAll {total} AEIoU grid runs complete (or skipped).")
"""
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 9 — Results Collection
    # ─────────────────────────────────────────────────────────────────────────
    cells.append(md(
"""## Section 9 · Results Collection

Ultralytics writes a `results.csv` file into each run directory after training.
This section collects all CSVs into a single flat DataFrame tagged with
`loss`, `split`, `seed`, and `rigidity` columns, then caches it to
`experiments_kvasir/all_results_combined.csv`.

**All downstream analysis cells load from this cache** — they never trigger
re-training. This means you can re-run Sections 10–13 on a fresh Colab session
(after uploading the cache CSV) without re-training any models.

**What to look for:** The combined DataFrame should have:
- One row per (run, epoch) combination
- No NaN in `metrics/mAP50-95(B)` at the final epoch
- `n_runs` matches the total planned runs
"""
    ))

    cells.append(code(
"""# --- Load all results.csv files into a single flat DataFrame
import pandas as pd

CACHE_CSV = EXPERIMENTS / "all_results_combined.csv"

def load_all_results(force_rebuild=False):
    # Returns DataFrame with all training metrics tagged by loss, split, seed.
    if CACHE_CSV.exists() and not force_rebuild:
        print(f"Loading from cache: {CACHE_CSV}")
        return pd.read_csv(CACHE_CSV)

    print("Building combined results from individual CSVs...")
    dfs = []

    for loss_name in ALL_LOSS_KEYS:
        for split_name in SPLIT_CONFIGS:
            for seed in SEEDS:
                run_name = f"kvasir_yolo26n_{loss_name}_{split_name}_s{seed}_e{EPOCHS}"
                csv_path = EXPERIMENTS / run_name / "results.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    df.columns = df.columns.str.strip()
                    df["run_name"] = run_name
                    df["loss"]     = loss_name
                    df["split"]    = split_name
                    df["seed"]     = seed
                    if "aeiou" in loss_name:
                        r_str = loss_name.split("_r")[1].replace("p",".")
                        df["rigidity"] = float(r_str)
                    else:
                        df["rigidity"] = -1.0
                    dfs.append(df)
                else:
                    print(f"  [MISSING] {csv_path}")

    if not dfs:
        raise RuntimeError("No results found. Run training cells first.")

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(CACHE_CSV, index=False)
    print(f"Saved combined CSV: {CACHE_CSV}  ({len(combined)} rows)")
    return combined


df_all = load_all_results()
print(f"\\nCombined DataFrame shape: {df_all.shape}")
print(f"Unique runs: {df_all['run_name'].nunique()}")
print(f"Losses found: {sorted(df_all['loss'].unique())}")
print(f"Seeds:  {sorted(df_all['seed'].unique())}")
"""
    ))

    # Analysis cells are in gen_03c.py
    return cells

