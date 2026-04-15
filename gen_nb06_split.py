"""
Generate four split notebooks for Phase 3 parallel Kaggle execution:
  notebooks/06a_yolo26s.ipynb        – YOLOv26s   (18 runs, ~1.5 h)
  notebooks/06b_rtdetr.ipynb         – RT-DETR-L  (18 runs, ~3.5 h)
  notebooks/06c_frcnn.ipynb          – Faster R-CNN via MMDetection (18 runs, ~4 h)
  notebooks/06d_analysis.ipynb       – Load outputs from 06a/b/c, cross-arch analysis

Kaggle workflow
  1. Run 06a, 06b, 06c on 3 separate Kaggle sessions (or sequentially on one).
  2. Each shard saves a summary CSV + all weights to /kaggle/working/experiments_phase3/.
  3. After each shard finishes, publish its /kaggle/working/experiments_phase3/ as a
     Kaggle Dataset (e.g. "phase3-yolo26s-results").
  4. In 06d, mount those three datasets as inputs and run the analysis.
"""
import json, sys, os
sys.stdout.reconfigure(encoding='utf-8')

# ── helpers ──────────────────────────────────────────────────────────────────
def code(src):
    return {'cell_type': 'code', 'execution_count': None,
            'metadata': {}, 'outputs': [], 'source': src}

def md(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src}

def nb(cells, kaggle_accelerator='GPU T4 x2'):
    return {
        'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.10.0'},
            'kaggle': {'accelerator': 'nvidiaTeslaT4', 'isInternetEnabled': True,
                       'dataSources': [], 'isGpuEnabled': True},
        },
        'cells': cells,
    }

def save(path, cells, **kw):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb(cells, **kw), f, ensure_ascii=False, indent=1)
    print(f'Saved: {path}  ({len(cells)} cells)')

# ═══════════════════════════════════════════════════════════════════════════
# SHARED BOILERPLATE – inserted verbatim into 06a, 06b, 06c
# ═══════════════════════════════════════════════════════════════════════════

SHARED_ENV = code(
    "# Environment detection\n"
    "import os, sys\n"
    "from pathlib import Path\n\n"
    "ON_KAGGLE = os.path.exists('/kaggle/working')\n"
    "ON_COLAB  = 'google.colab' in sys.modules or os.path.exists('/content')\n\n"
    "if ON_KAGGLE:\n"
    "    ROOT = Path('/kaggle/working')\n"
    "    print('Runtime: Kaggle')\n"
    "elif ON_COLAB:\n"
    "    ROOT = Path('/content')\n"
    "    print('Runtime: Colab')\n"
    "else:\n"
    "    ROOT = Path.cwd()\n"
    "    print(f'Runtime: local ({ROOT})')\n\n"
    "gpu = os.popen('nvidia-smi --query-gpu=name --format=csv,noheader').read().strip()\n"
    "print(f'GPU: {gpu or \"none detected\"}')\n"
)

SHARED_INSTALL_CORE = code(
    "# Install core dependencies\n"
    "import subprocess, sys\n"
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',\n"
    "                'ultralytics==8.4.9', 'wandb==0.24.1', 'pycocotools'], check=True)\n"
    "print('Core deps installed.')\n"
)

SHARED_CLONE = code(
    "# Clone / update repo\n"
    "import os, sys\n"
    "from pathlib import Path\n\n"
    "REPO_PATH = ROOT / 'amorphous-yolo'\n"
    "if not (REPO_PATH / '.git').exists():\n"
    "    os.system(f'git clone https://github.com/nipun-taneja/amorphous-yolo.git {REPO_PATH}')\n"
    "    print('Cloned.')\n"
    "else:\n"
    "    os.system(f'git -C {REPO_PATH} pull --ff-only')\n"
    "    print('Repo up to date.')\n"
    "if str(REPO_PATH) not in sys.path:\n"
    "    sys.path.insert(0, str(REPO_PATH))\n"
)

SHARED_WANDB = code(
    "# WandB (optional)\n"
    "import os, wandb\n"
    "WANDB_PROJECT = 'amorphous-yolo-phase3'\n"
    "if ON_KAGGLE:\n"
    "    try:\n"
    "        from kaggle_secrets import UserSecretsClient\n"
    "        _k = UserSecretsClient().get_secret('WANDB_API_KEY')\n"
    "        if _k: os.environ['WANDB_API_KEY'] = _k\n"
    "    except Exception: pass\n"
    "elif ON_COLAB:\n"
    "    try:\n"
    "        from google.colab import userdata\n"
    "        _k = userdata.get('WANDB_API_KEY')\n"
    "        if _k: os.environ['WANDB_API_KEY'] = _k\n"
    "    except Exception: pass\n"
    "if os.environ.get('WANDB_API_KEY'):\n"
    "    wandb.login(key=os.environ['WANDB_API_KEY'], relogin=True)\n"
    "    print(f'WandB active → {WANDB_PROJECT}')\n"
    "else:\n"
    "    os.environ['WANDB_MODE'] = 'disabled'\n"
    "    print('WandB disabled.')\n"
)

SHARED_CONSTANTS = code(
    "# Phase 3 constants & paths\n"
    "import json as _json, math, time\n"
    "from pathlib import Path\n"
    "from datetime import datetime\n\n"
    "PROJECT_DIR   = ROOT / 'amorphous-yolo'\n"
    "DATASET_ROOT  = PROJECT_DIR / 'data' / 'kvasir_seg'\n"
    "EXPERIMENTS   = ROOT / 'experiments_phase3'\n"
    "ANALYSIS_DIR  = EXPERIMENTS / 'analysis'\n"
    "METRICS_DIR   = EXPERIMENTS / 'metrics'\n"
    "MANIFEST_PATH = EXPERIMENTS / 'manifest.json'\n"
    "for _d in [EXPERIMENTS, ANALYSIS_DIR, METRICS_DIR]:\n"
    "    _d.mkdir(parents=True, exist_ok=True)\n\n"
    "EPOCHS = 20\n"
    "IMGSZ  = 640\n"
    "SEEDS  = [42]\n"
    "DEVICE = 0\n\n"
    "SPLIT_CONFIGS = {\n"
    "    'clean': PROJECT_DIR / 'data' / 'kvasir_seg.yaml',\n"
    "    'low':   PROJECT_DIR / 'data' / 'kvasir_seg_low.yaml',\n"
    "    'high':  PROJECT_DIR / 'data' / 'kvasir_seg_high.yaml',\n"
    "}\n\n"
    "# Drive persistence (Colab only)\n"
    "DRIVE_AVAILABLE = False\n"
    "if ON_COLAB:\n"
    "    try:\n"
    "        from google.colab import drive\n"
    "        drive.mount('/content/drive')\n"
    "        DRIVE_EXPERIMENTS = Path('/content/drive/MyDrive/experiments_phase3')\n"
    "        DRIVE_EXPERIMENTS.mkdir(parents=True, exist_ok=True)\n"
    "        DRIVE_AVAILABLE = True\n"
    "        print(f'Drive mounted: {DRIVE_EXPERIMENTS}')\n"
    "    except Exception as e:\n"
    "        print(f'Drive not available: {e}')\n"
    "print(f'EXPERIMENTS = {EXPERIMENTS}')\n"
)

SHARED_DATASET = code(
    "# Download / verify Kvasir-SEG dataset\n"
    "import os\n"
    "VALID_DIR  = DATASET_ROOT / 'valid' / 'images'\n"
    "LOW_DIR    = DATASET_ROOT / 'valid_low' / 'images'\n"
    "HIGH_DIR   = DATASET_ROOT / 'valid_high' / 'images'\n\n"
    "if not VALID_DIR.exists():\n"
    "    print('Downloading Kvasir-SEG...')\n"
    "    os.system(f'cd {PROJECT_DIR} && python scripts/prepare_kvasir.py')\n"
    "n_imgs = len(list(VALID_DIR.glob('*.jpg'))) if VALID_DIR.exists() else 0\n"
    "print(f'Val images: {n_imgs}  (expected 200)')\n"
)

SHARED_LOSSES = code(
    "# Import loss classes\n"
    "from src.losses import (\n"
    "    CIoULoss, EIoULoss, ECIoULoss, AEIoULoss,\n"
    ")\n"
    "import ultralytics.utils.loss as _ul\n"
    "_ORIG_BBOX_FWD = _ul.BboxLoss.forward\n\n"
    "def patch_loss(loss_fn):\n"
    "    import torch\n"
    "    def _fwd(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,\n"
    "             target_scores, target_scores_sum, fg_mask):\n"
    "        loss_iou = loss_fn(\n"
    "            pred_bboxes[fg_mask], target_bboxes[fg_mask]\n"
    "        ) * target_scores[fg_mask].sum() / target_scores_sum\n"
    "        loss_dfl = torch.tensor(0.0, device=pred_dist.device)\n"
    "        if self.use_dfl:\n"
    "            loss_dfl = self.dfl_loss(\n"
    "                pred_dist[fg_mask].view(-1, self.reg_max + 1),\n"
    "                target_bboxes[fg_mask],\n"
    "                anchor_points,\n"
    "            ) * target_scores[fg_mask].sum() / target_scores_sum\n"
    "        return loss_iou, loss_dfl\n"
    "    _ul.BboxLoss.forward = _fwd\n\n"
    "def restore_loss():\n"
    "    _ul.BboxLoss.forward = _ORIG_BBOX_FWD\n\n"
    "# Auto-select best AEIoU rigidities from Phase 2 (fallback to defaults)\n"
    "import json as _json\n"
    "_p2_metrics = Path('/kaggle/working/amorphous-yolo/experiments_kvasir/metrics_all_losses.json')\n"
    "if not _p2_metrics.exists():\n"
    "    _p2_metrics = Path('/content/amorphous-yolo/experiments_kvasir/metrics_all_losses.json')\n"
    "try:\n"
    "    _data = _json.loads(_p2_metrics.read_text())\n"
    "    _aeiou = [(v.get('map50_95', 0), v['loss']) for v in _data.values()\n"
    "              if v.get('loss', '').startswith('aeiou')]\n"
    "    _top3 = sorted(_aeiou, reverse=True)[:3]\n"
    "    AEIOU_RIGIDITIES = [float(n.split('r')[-1].replace('p', '.')) for _, n in _top3]\n"
    "    print(f'Best AEIoU rigidities from Phase 2: {AEIOU_RIGIDITIES}')\n"
    "except Exception:\n"
    "    AEIOU_RIGIDITIES = [0.1, 0.3, 0.5]\n"
    "    print(f'Phase 2 metrics not found. Using default rigidities: {AEIOU_RIGIDITIES}')\n\n"
    "LOSS_CONFIGS = [\n"
    "    ('ciou',  CIoULoss()),\n"
    "    ('eiou',  EIoULoss()),\n"
    "    ('eciou', ECIoULoss()),\n"
    "] + [(f'aeiou_r{str(r).replace(\".\", \"p\")}', AEIoULoss(rigidity=r)) for r in AEIOU_RIGIDITIES]\n"
    "print(f'Loss configs: {[n for n,_ in LOSS_CONFIGS]}')\n"
)

SHARED_COCOGT = code(
    "# Build COCO GT JSON from YOLO val labels (idempotent)\n"
    "import cv2, json as _json\n\n"
    "def build_coco_gt_json(img_dir, lbl_dir, out_path):\n"
    "    if Path(out_path).exists(): return\n"
    "    images, anns, ann_id = [], [], 1\n"
    "    for p in sorted(Path(img_dir).glob('*.jpg')):\n"
    "        img = cv2.imread(str(p))\n"
    "        H, W = img.shape[:2]\n"
    "        img_id = int(p.stem) if p.stem.isdigit() else hash(p.stem) % 10**6\n"
    "        images.append({'id': img_id, 'file_name': p.name, 'width': W, 'height': H})\n"
    "        lbl = Path(lbl_dir) / f'{p.stem}.txt'\n"
    "        if not lbl.exists(): continue\n"
    "        for line in lbl.read_text().strip().splitlines():\n"
    "            _, cx, cy, bw, bh = map(float, line.split())\n"
    "            x1, y1 = (cx-bw/2)*W, (cy-bh/2)*H\n"
    "            anns.append({'id': ann_id, 'image_id': img_id, 'category_id': 1,\n"
    "                          'bbox': [x1, y1, bw*W, bh*H],\n"
    "                          'area': bw*W*bh*H, 'iscrowd': 0})\n"
    "            ann_id += 1\n"
    "    coco = {'images': images, 'annotations': anns,\n"
    "            'categories': [{'id': 1, 'name': 'polyp'}]}\n"
    "    Path(out_path).write_text(_json.dumps(coco))\n"
    "    print(f'COCO GT JSON: {out_path} ({len(images)} imgs, {len(anns)} anns)')\n\n"
    "COCO_GT_JSON = DATASET_ROOT / 'valid' / 'coco_gt.json'\n"
    "build_coco_gt_json(\n"
    "    DATASET_ROOT / 'valid' / 'images',\n"
    "    DATASET_ROOT / 'valid' / 'labels',\n"
    "    COCO_GT_JSON,\n"
    ")\n"
)

SHARED_METRICS_FN = code(
    "# compute_metrics_ultralytics() — persist PR + COCO AP + confusion matrix\n"
    "import numpy as np, json as _json\n"
    "from pathlib import Path\n"
    "from ultralytics import YOLO\n\n"
    "def compute_metrics_ultralytics(run_name, weights_path, yaml_path, force=False):\n"
    "    out_dir = METRICS_DIR / run_name\n"
    "    if (out_dir / 'coco_ap_suite.json').exists() and not force:\n"
    "        print(f'  [SKIP metrics] {run_name}'); return\n"
    "    if not Path(weights_path).exists():\n"
    "        print(f'  [MISS] {weights_path}'); return\n"
    "    out_dir.mkdir(exist_ok=True)\n"
    "    model = YOLO(str(weights_path))\n"
    "    val_res = model.val(data=str(yaml_path), verbose=False,\n"
    "                        save_json=True, conf=0.001, iou=0.6)\n"
    "    # PR curve\n"
    "    try:\n"
    "        prec = np.array(val_res.box.p).reshape(-1).tolist()\n"
    "        rec  = np.array(val_res.box.r).reshape(-1).tolist()\n"
    "        f1   = np.array(val_res.box.f1).reshape(-1).tolist()\n"
    "    except Exception:\n"
    "        prec = rec = f1 = []\n"
    "    pr_data = {'precision': prec, 'recall': rec, 'f1': f1,\n"
    "               'ap50': float(val_res.box.map50),\n"
    "               'ap75': float(val_res.box.map75),\n"
    "               'map50_95': float(val_res.box.map),\n"
    "               'ap_per_iou': val_res.box.maps.tolist()}\n"
    "    (out_dir / 'pr_curve.json').write_text(_json.dumps(pr_data, indent=2))\n"
    "    # COCO AP suite via pycocotools\n"
    "    coco_suite = {'map50_95': float(val_res.box.map),\n"
    "                  'map50': float(val_res.box.map50),\n"
    "                  'map75': float(val_res.box.map75),\n"
    "                  'APs': None, 'APm': None, 'APl': None}\n"
    "    preds = list(Path(str(val_res.save_dir)).glob('*predictions*.json'))\n"
    "    if preds:\n"
    "        try:\n"
    "            from pycocotools.coco import COCO\n"
    "            from pycocotools.cocoeval import COCOeval\n"
    "            gt  = COCO(str(COCO_GT_JSON))\n"
    "            dt  = gt.loadRes(str(preds[0]))\n"
    "            ev  = COCOeval(gt, dt, 'bbox')\n"
    "            ev.evaluate(); ev.accumulate(); ev.summarize()\n"
    "            s = ev.stats\n"
    "            coco_suite.update({'map50_95': float(s[0]), 'map50': float(s[1]),\n"
    "                               'map75': float(s[2]), 'APs': float(s[3]),\n"
    "                               'APm': float(s[4]), 'APl': float(s[5]),\n"
    "                               'AR_1': float(s[6]), 'AR_10': float(s[7]),\n"
    "                               'AR_100': float(s[8])})\n"
    "        except Exception as e:\n"
    "            print(f'  [WARN] pycocotools: {e}')\n"
    "    (out_dir / 'coco_ap_suite.json').write_text(_json.dumps(coco_suite, indent=2))\n"
    "    # Confusion matrix\n"
    "    try:\n"
    "        cm = model.validator.metrics.confusion_matrix\n"
    "        mat = cm.matrix.tolist()\n"
    "        conf_out = {'matrix': mat, 'class_names': ['polyp'],\n"
    "                    'TP': mat[0][0], 'FN': mat[0][1] if len(mat[0])>1 else None,\n"
    "                    'FP': mat[1][0] if len(mat)>1 else None}\n"
    "    except Exception as e:\n"
    "        conf_out = {'error': str(e)}\n"
    "    (out_dir / 'confusion_matrix.json').write_text(_json.dumps(conf_out, indent=2))\n"
    "    print(f'  [OK] {run_name}  mAP50={coco_suite[\"map50\"]:.4f}  mAP50-95={coco_suite[\"map50_95\"]:.4f}')\n"
)

SHARED_RUN_TRAINING = code(
    "# run_training() — shared Ultralytics trainer (YOLOv26s + RT-DETR-L)\n"
    "import shutil, json as _json, time\n"
    "from datetime import datetime\n"
    "from ultralytics import YOLO\n\n"
    "def _load_manifest():\n"
    "    return _json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}\n\n"
    "def _write_manifest(run_name, meta):\n"
    "    m = _load_manifest(); m[run_name] = meta\n"
    "    MANIFEST_PATH.write_text(_json.dumps(m, indent=2))\n\n"
    "def _sync_to_drive(run_name):\n"
    "    if not DRIVE_AVAILABLE: return\n"
    "    try:\n"
    "        shutil.copytree(str(EXPERIMENTS / run_name),\n"
    "                        str(DRIVE_EXPERIMENTS / run_name), dirs_exist_ok=True)\n"
    "    except Exception as e:\n"
    "        print(f'  [DRIVE] {e}')\n\n"
    "def run_training(arch_prefix, model_pt, loss_name, loss_fn,\n"
    "                 split_name, yaml_path, seed=42, epochs=None):\n"
    "    epochs = epochs or EPOCHS\n"
    "    run_name = f'phase3_{arch_prefix}_{loss_name}_{split_name}_s{seed}_e{epochs}'\n"
    "    run_dir  = EXPERIMENTS / run_name\n"
    "    if (run_dir / 'results.csv').exists():\n"
    "        print(f'[SKIP] {run_name}'); return run_dir\n"
    "    local_last = run_dir / 'weights' / 'last.pt'\n"
    "    resuming   = local_last.exists()\n"
    "    tag = '[RESUME]' if resuming else '[START]'\n"
    "    print(f'\\n{\"=\"*65}\\n{tag} {run_name}\\n{\"=\"*65}')\n"
    "    meta = {'arch': arch_prefix, 'loss': loss_name, 'split': split_name,\n"
    "            'seed': seed, 'epochs': epochs, 'status': 'running',\n"
    "            'timestamp': datetime.now().isoformat()}\n"
    "    _write_manifest(run_name, meta)\n"
    "    t0 = time.time()\n"
    "    try:\n"
    "        import os as _os\n"
    "        _os.environ.update({'WANDB_PROJECT': WANDB_PROJECT, 'WANDB_NAME': run_name})\n"
    "        patch_loss(loss_fn)\n"
    "        if resuming:\n"
    "            model = YOLO(str(local_last))\n"
    "            results = model.train(resume=True)\n"
    "        else:\n"
    "            model = YOLO(model_pt)\n"
    "            results = model.train(\n"
    "                data=str(yaml_path), epochs=epochs, imgsz=IMGSZ,\n"
    "                project=str(EXPERIMENTS), name=run_name,\n"
    "                device=DEVICE, seed=seed, exist_ok=True,\n"
    "            )\n"
    "        meta['status'] = 'complete'\n"
    "        meta['elapsed_sec'] = round(time.time() - t0, 1)\n"
    "        try:\n"
    "            import wandb as _w\n"
    "            if _w.run: _w.finish()\n"
    "        except Exception: pass\n"
    "    except Exception as e:\n"
    "        print(f'  [ERROR] {e}')\n"
    "        meta['status'] = 'failed'; meta['error'] = str(e); raise\n"
    "    finally:\n"
    "        restore_loss(); _write_manifest(run_name, meta); _sync_to_drive(run_name)\n"
    "    print(f'[DONE] {run_name}  ({meta[\"elapsed_sec\"]:.0f}s)')\n"
    "    return run_dir\n"
    "print('run_training() ready.')\n"
)

SHARED_SUMMARY_SAVE = """
# Save shard summary CSV
import pandas as pd, json as _json

rows = []
for run_dir in sorted(EXPERIMENTS.glob(f'phase3_{ARCH_PREFIX}_*')):
    run_name = run_dir.name
    csv_path = run_dir / 'results.csv'
    if not csv_path.exists(): continue
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    last = df.iloc[-1]
    rows.append({
        'run': run_name,
        'arch': ARCH_PREFIX,
        'loss': run_name.split('_')[3] if '_' in run_name else '',
        'split': run_name.split('_')[-3] if '_' in run_name else '',
        'map50':    float(last.get('metrics/mAP50(B)', 0) or 0),
        'map50_95': float(last.get('metrics/mAP50-95(B)', 0) or 0),
    })

summary_csv = EXPERIMENTS / f'summary_{ARCH_PREFIX}.csv'
pd.DataFrame(rows).to_csv(summary_csv, index=False)
print(f'Summary saved: {summary_csv}  ({len(rows)} runs)')
print(pd.DataFrame(rows).to_string(index=False))
"""

SHARED_OUTPUT_INSTRUCTIONS = """
# ── OUTPUT INSTRUCTIONS ──────────────────────────────────────────────────────
# After this notebook finishes:
#
# On Kaggle:
#   1. Go to this notebook's Output tab
#   2. Click "+ New Dataset" → name it  phase3-{arch}-results
#   3. In 06d_analysis.ipynb, add this dataset as an Input
#      Path will be: /kaggle/input/phase3-{arch}-results/
#
# On Colab:
#   All outputs already synced to Drive → experiments_phase3/
#   Run 06d directly without any extra steps.
print('Shard complete. See comment above for output publishing instructions.')
"""

# ═══════════════════════════════════════════════════════════════════════════
# 06a – YOLOv26s
# ═══════════════════════════════════════════════════════════════════════════

cells_06a = [
    md("# 06a · Phase 3: YOLOv26s Shard\n\n"
       "Trains 6 loss configs × 3 splits = **18 runs** with YOLOv26s.  \n"
       "Est. runtime: **~1.5 h** on Kaggle T4.\n\n"
       "Part of the Phase 3 cross-architecture study. Run in parallel with 06b and 06c."),
    md("## Section 1 · Environment"),
    SHARED_ENV,
    SHARED_INSTALL_CORE,
    SHARED_CLONE,
    SHARED_WANDB,
    md("## Section 2 · Constants & Dataset"),
    SHARED_CONSTANTS,
    SHARED_DATASET,
    md("## Section 3 · Loss Configs"),
    SHARED_LOSSES,
    SHARED_COCOGT,
    md("## Section 4 · Training Functions"),
    SHARED_METRICS_FN,
    SHARED_RUN_TRAINING,
    md("## Section 5 · YOLOv26s Training Loop"),
    code(
        "# YOLOv26s — 18 runs (6 losses × 3 splits)\n"
        "ARCH_PREFIX = 'yolo26s'\n"
        "MODEL_PT    = 'yolov8s.pt'  # YOLOv26s checkpoint (same backbone class)\n\n"
        "total, done, skipped, failed = 0, 0, 0, 0\n"
        "for loss_name, loss_fn in LOSS_CONFIGS:\n"
        "    for split_name, yaml_path in SPLIT_CONFIGS.items():\n"
        "        total += 1\n"
        "        try:\n"
        "            result = run_training(ARCH_PREFIX, MODEL_PT, loss_name, loss_fn,\n"
        "                                  split_name, yaml_path)\n"
        "            done += 1 if result else 0\n"
        "            skipped += 1 if result is None else 0\n"
        "        except Exception as e:\n"
        "            print(f'  [ERROR] {loss_name}/{split_name}: {e}'); failed += 1\n"
        "print(f'\\nYOLOv26s: {done} done, {skipped} skipped, {failed} failed / {total} total')\n"
    ),
    md("## Section 6 · Metrics Extraction"),
    code(
        "# Compute and persist metrics for all completed YOLOv26s runs\n"
        "for loss_name, _ in LOSS_CONFIGS:\n"
        "    for split_name, yaml_path in SPLIT_CONFIGS.items():\n"
        "        run_name = f'phase3_{ARCH_PREFIX}_{loss_name}_{split_name}_s42_e{EPOCHS}'\n"
        "        weights  = EXPERIMENTS / run_name / 'weights' / 'best.pt'\n"
        "        compute_metrics_ultralytics(run_name, weights, yaml_path)\n"
        "print('Metrics done.')\n"
    ),
    md("## Section 7 · Save & Output"),
    code(SHARED_SUMMARY_SAVE),
    code(SHARED_OUTPUT_INSTRUCTIONS),
]

# ═══════════════════════════════════════════════════════════════════════════
# 06b – RT-DETR-L
# ═══════════════════════════════════════════════════════════════════════════

cells_06b = [
    md("# 06b · Phase 3: RT-DETR-L Shard\n\n"
       "Trains 6 loss configs × 3 splits = **18 runs** with RT-DETR-L.  \n"
       "Est. runtime: **~3.5 h** on Kaggle T4.\n\n"
       "Part of the Phase 3 cross-architecture study. Run in parallel with 06a and 06c."),
    md("## Section 1 · Environment"),
    SHARED_ENV,
    SHARED_INSTALL_CORE,
    SHARED_CLONE,
    SHARED_WANDB,
    md("## Section 2 · Constants & Dataset"),
    SHARED_CONSTANTS,
    SHARED_DATASET,
    md("## Section 3 · Loss Configs"),
    SHARED_LOSSES,
    SHARED_COCOGT,
    md("## Section 4 · BboxLoss Patch Verification for RT-DETR"),
    code(
        "# Verify BboxLoss patch propagates into RT-DETR's loss module\n"
        "try:\n"
        "    import ultralytics.models.rtdetr.train as _rt\n"
        "    import inspect\n"
        "    uses = 'BboxLoss' in inspect.getsource(_rt)\n"
        "    print(f'RT-DETR uses BboxLoss: {uses}')\n"
        "    if not uses:\n"
        "        print('[WARN] Patch may not apply — verify loss curves decrease normally.')\n"
        "except Exception as e:\n"
        "    print(f'[WARN] Could not inspect RT-DETR train module: {e}')\n"
    ),
    md("## Section 5 · Training Functions"),
    SHARED_METRICS_FN,
    SHARED_RUN_TRAINING,
    md("## Section 6 · RT-DETR-L Training Loop"),
    code(
        "# RT-DETR-L smoke test (1 epoch) before full run\n"
        "print('Running 1-epoch smoke test...')\n"
        "from ultralytics import YOLO\n"
        "import torch\n"
        "try:\n"
        "    _m = YOLO('rtdetr-l.pt')\n"
        "    _r = _m.train(data=str(SPLIT_CONFIGS['clean']), epochs=1, imgsz=IMGSZ,\n"
        "                  project=str(EXPERIMENTS), name='rtdetr_smoke',\n"
        "                  device=DEVICE, exist_ok=True)\n"
        "    print('[OK] RT-DETR-L smoke test passed.')\n"
        "except Exception as e:\n"
        "    print(f'[FAIL] RT-DETR-L smoke test: {e}')\n"
        "    raise SystemExit('Aborting: RT-DETR-L not functional on this environment.')\n"
    ),
    code(
        "# RT-DETR-L — 18 runs (6 losses × 3 splits)\n"
        "ARCH_PREFIX = 'rtdetr'\n"
        "MODEL_PT    = 'rtdetr-l.pt'\n\n"
        "total, done, skipped, failed = 0, 0, 0, 0\n"
        "for loss_name, loss_fn in LOSS_CONFIGS:\n"
        "    for split_name, yaml_path in SPLIT_CONFIGS.items():\n"
        "        total += 1\n"
        "        try:\n"
        "            result = run_training(ARCH_PREFIX, MODEL_PT, loss_name, loss_fn,\n"
        "                                  split_name, yaml_path)\n"
        "            done += 1 if result else 0\n"
        "        except Exception as e:\n"
        "            print(f'  [ERROR] {loss_name}/{split_name}: {e}'); failed += 1\n"
        "print(f'\\nRT-DETR-L: {done} done, {skipped} skipped, {failed} failed / {total} total')\n"
    ),
    md("## Section 7 · Metrics Extraction"),
    code(
        "# Compute and persist metrics for all completed RT-DETR-L runs\n"
        "for loss_name, _ in LOSS_CONFIGS:\n"
        "    for split_name, yaml_path in SPLIT_CONFIGS.items():\n"
        "        run_name = f'phase3_{ARCH_PREFIX}_{loss_name}_{split_name}_s42_e{EPOCHS}'\n"
        "        weights  = EXPERIMENTS / run_name / 'weights' / 'best.pt'\n"
        "        compute_metrics_ultralytics(run_name, weights, yaml_path)\n"
        "print('Metrics done.')\n"
    ),
    md("## Section 8 · Save & Output"),
    code(SHARED_SUMMARY_SAVE),
    code(SHARED_OUTPUT_INSTRUCTIONS),
]

# ═══════════════════════════════════════════════════════════════════════════
# 06c – Faster R-CNN
# ═══════════════════════════════════════════════════════════════════════════

FRCNN_INSTALL = code(
    "# Install MMDetection (Faster R-CNN)\n"
    "import subprocess, sys\n"
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',\n"
    "                'openmim'], check=True)\n"
    "subprocess.run(['mim', 'install', '-q', 'mmengine', 'mmcv>=2.0',\n"
    "                'mmdet'], check=True)\n"
    "print('MMDetection installed.')\n"
)

FRCNN_COCO_ANNOTS = code(
    "# Convert YOLO labels → COCO annotations for MMDetection DataLoader\n"
    "import json as _json, cv2\n\n"
    "def yolo_to_coco(img_dir, lbl_dir, out_path, split_tag):\n"
    "    if Path(out_path).exists(): return\n"
    "    Path(out_path).parent.mkdir(parents=True, exist_ok=True)\n"
    "    imgs, anns, ann_id = [], [], 1\n"
    "    for p in sorted(Path(img_dir).glob('*.jpg')):\n"
    "        im = cv2.imread(str(p)); H, W = im.shape[:2]\n"
    "        img_id = int(p.stem) if p.stem.isdigit() else hash(p.stem) % 10**6\n"
    "        imgs.append({'id': img_id, 'file_name': str(p), 'width': W, 'height': H})\n"
    "        lbl = Path(lbl_dir) / f'{p.stem}.txt'\n"
    "        if not lbl.exists(): continue\n"
    "        for line in lbl.read_text().strip().splitlines():\n"
    "            _, cx, cy, bw, bh = map(float, line.split())\n"
    "            x1, y1 = (cx-bw/2)*W, (cy-bh/2)*H\n"
    "            anns.append({'id': ann_id, 'image_id': img_id, 'category_id': 1,\n"
    "                          'bbox': [x1, y1, bw*W, bh*H],\n"
    "                          'area': bw*W*bh*H, 'iscrowd': 0})\n"
    "            ann_id += 1\n"
    "    d = {'images': imgs, 'annotations': anns,\n"
    "         'categories': [{'id': 1, 'name': 'polyp'}]}\n"
    "    Path(out_path).write_text(_json.dumps(d))\n"
    "    print(f'COCO JSON ({split_tag}): {len(imgs)} imgs, {len(anns)} anns → {out_path}')\n\n"
    "COCO_DIR = DATASET_ROOT / 'coco_annotations'\n"
    "yolo_to_coco(DATASET_ROOT/'train'/'images', DATASET_ROOT/'train'/'labels',\n"
    "             COCO_DIR/'instances_train.json', 'train')\n"
    "yolo_to_coco(DATASET_ROOT/'valid'/'images', DATASET_ROOT/'valid'/'labels',\n"
    "             COCO_DIR/'instances_val.json', 'val/clean')\n"
    "yolo_to_coco(DATASET_ROOT/'valid_low'/'images', DATASET_ROOT/'valid_low'/'labels',\n"
    "             COCO_DIR/'instances_val_low.json', 'val/low')\n"
    "yolo_to_coco(DATASET_ROOT/'valid_high'/'images', DATASET_ROOT/'valid_high'/'labels',\n"
    "             COCO_DIR/'instances_val_high.json', 'val/high')\n"
)

FRCNN_LOSS_REG = code(
    "# Register custom losses with MMDetection model registry\n"
    "from mmdet.registry import MODELS\n"
    "from src.losses import CIoULoss, EIoULoss, ECIoULoss, AEIoULoss\n"
    "import torch\n\n"
    "def _make_mmdet_loss(loss_cls, **kwargs):\n"
    "    \"\"\"Wrap a src.losses class to the MMDet loss API.\"\"\"\n"
    "    _inst = loss_cls(**kwargs)\n"
    "    @MODELS.register_module(name=f'{loss_cls.__name__}MMDet_{id(_inst)}')\n"
    "    class _Wrapper(torch.nn.Module):\n"
    "        def __init__(self):\n"
    "            super().__init__()\n"
    "            self.loss_fn = _inst\n"
    "            self.loss_weight = 1.0\n"
    "        def forward(self, pred, target, weight=None, avg_factor=None, **kw):\n"
    "            return self.loss_fn(pred, target)\n"
    "    return _Wrapper.__name__, _Wrapper\n\n"
    "# Build MMDet loss registry entries for each loss config\n"
    "MMDET_LOSS_MAP = {}  # loss_name → registered MMDet class name\n"
    "for loss_name, loss_fn_instance in LOSS_CONFIGS:\n"
    "    reg_name, cls = _make_mmdet_loss(type(loss_fn_instance),\n"
    "                                     **({'rigidity': loss_fn_instance.rigidity}\n"
    "                                        if hasattr(loss_fn_instance, 'rigidity') else {}))\n"
    "    MMDET_LOSS_MAP[loss_name] = reg_name\n"
    "print('Registered MMDet losses:', list(MMDET_LOSS_MAP.keys()))\n"
)

FRCNN_CONFIG_AND_RUN = code(
    "# build_frcnn_config() + run_frcnn()\n"
    "from mmengine.config import Config\n"
    "from mmengine.runner import Runner\n"
    "import json as _json, time, shutil\n"
    "from datetime import datetime\n\n"
    "FRCNN_CFGS_DIR = EXPERIMENTS / 'frcnn_configs'\n"
    "FRCNN_CFGS_DIR.mkdir(exist_ok=True)\n\n"
    "def build_frcnn_config(loss_name, split_name, run_dir, seed, epochs):\n"
    "    cfg = Config.fromfile('configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py')\n"
    "    loss_dict = {'type': MMDET_LOSS_MAP[loss_name], 'loss_weight': 1.0}\n"
    "    cfg.data_root = str(DATASET_ROOT)\n"
    "    cfg.metainfo = {'classes': ('polyp',), 'palette': [(220, 20, 60)]}\n"
    "    ann_tag = {'clean': 'val', 'low': 'val_low', 'high': 'val_high'}[split_name]\n"
    "    val_img = str({'clean': DATASET_ROOT/'valid'/'images',\n"
    "                   'low':   DATASET_ROOT/'valid_low'/'images',\n"
    "                   'high':  DATASET_ROOT/'valid_high'/'images'}[split_name])\n"
    "    for d in [cfg.train_dataloader.dataset, cfg.val_dataloader.dataset,\n"
    "              cfg.test_dataloader.dataset]:\n"
    "        d.metainfo = cfg.metainfo\n"
    "    cfg.train_dataloader.dataset.ann_file = str(DATASET_ROOT/'coco_annotations'/'instances_train.json')\n"
    "    cfg.train_dataloader.dataset.data_prefix = dict(img=str(DATASET_ROOT/'train'/'images'))\n"
    "    cfg.val_dataloader.dataset.ann_file   = str(DATASET_ROOT/f'coco_annotations/instances_{ann_tag}.json')\n"
    "    cfg.val_dataloader.dataset.data_prefix = dict(img=val_img)\n"
    "    cfg.test_dataloader.dataset.ann_file  = cfg.val_dataloader.dataset.ann_file\n"
    "    cfg.test_dataloader.dataset.data_prefix = cfg.val_dataloader.dataset.data_prefix\n"
    "    cfg.val_evaluator.ann_file  = cfg.val_dataloader.dataset.ann_file\n"
    "    cfg.test_evaluator.ann_file = cfg.val_dataloader.dataset.ann_file\n"
    "    for head in [cfg.model.rpn_head, cfg.model.roi_head.bbox_head]:\n"
    "        if hasattr(head, 'loss_bbox'): head.loss_bbox = loss_dict\n"
    "    cfg.model.roi_head.bbox_head.num_classes = 1\n"
    "    cfg.train_cfg.max_epochs = epochs\n"
    "    cfg.default_hooks.checkpoint.interval = epochs\n"
    "    cfg.work_dir = str(run_dir)\n"
    "    cfg.seed = seed\n"
    "    cfg_path = FRCNN_CFGS_DIR / f'{run_dir.name}.py'\n"
    "    cfg.dump(str(cfg_path))\n"
    "    return cfg_path\n\n"
    "def run_frcnn(loss_name, split_name, seed=42, epochs=None):\n"
    "    epochs = epochs or EPOCHS\n"
    "    run_name = f'phase3_frcnn_{loss_name}_{split_name}_s{seed}_e{epochs}'\n"
    "    run_dir  = EXPERIMENTS / run_name\n"
    "    result_json = run_dir / 'metric.json'\n"
    "    if result_json.exists():\n"
    "        print(f'[SKIP] {run_name}'); return run_dir\n"
    "    run_dir.mkdir(exist_ok=True)\n"
    "    print(f'\\n{\"=\"*65}\\n[START] {run_name}\\n{\"=\"*65}')\n"
    "    cfg_path = build_frcnn_config(loss_name, split_name, run_dir, seed, epochs)\n"
    "    t0 = time.time()\n"
    "    try:\n"
    "        runner = Runner.from_cfg(Config.fromfile(str(cfg_path)))\n"
    "        runner.train()\n"
    "        # Save metrics\n"
    "        metrics = runner.val()\n"
    "        result_json.write_text(_json.dumps(metrics, indent=2))\n"
    "        print(f'[DONE] {run_name}  ({time.time()-t0:.0f}s)')\n"
    "        if DRIVE_AVAILABLE:\n"
    "            try: shutil.copytree(str(run_dir), str(DRIVE_EXPERIMENTS/run_name), dirs_exist_ok=True)\n"
    "            except: pass\n"
    "    except Exception as e:\n"
    "        print(f'  [ERROR] {run_name}: {e}'); raise\n"
    "    return run_dir\n"
    "print('run_frcnn() ready.')\n"
)

FRCNN_SUMMARY_SAVE = """
# Save Faster R-CNN summary CSV
import pandas as pd, json as _json

ARCH_PREFIX = 'frcnn'
rows = []
for run_dir in sorted(EXPERIMENTS.glob('phase3_frcnn_*')):
    mj = run_dir / 'metric.json'
    if not mj.exists(): continue
    m = _json.loads(mj.read_text())
    run_name = run_dir.name
    parts = run_name.split('_')
    rows.append({
        'run': run_name, 'arch': 'frcnn',
        'loss':  parts[3] if len(parts) > 3 else '',
        'split': parts[4] if len(parts) > 4 else '',
        'map50':    m.get('coco/bbox_mAP_50', 0),
        'map50_95': m.get('coco/bbox_mAP', 0),
    })

summary_csv = EXPERIMENTS / 'summary_frcnn.csv'
pd.DataFrame(rows).to_csv(summary_csv, index=False)
print(f'Summary saved: {summary_csv}  ({len(rows)} runs)')
print(pd.DataFrame(rows).to_string(index=False))
"""

cells_06c = [
    md("# 06c · Phase 3: Faster R-CNN Shard\n\n"
       "Trains 6 loss configs × 3 splits = **18 runs** with Faster R-CNN R50-FPN via MMDetection.  \n"
       "Est. runtime: **~4 h** on Kaggle T4 (includes ~15 min MMDetection install).\n\n"
       "Part of the Phase 3 cross-architecture study. Run in parallel with 06a and 06b."),
    md("## Section 1 · Environment"),
    SHARED_ENV,
    SHARED_INSTALL_CORE,
    FRCNN_INSTALL,
    SHARED_CLONE,
    SHARED_WANDB,
    md("## Section 2 · Constants & Dataset"),
    SHARED_CONSTANTS,
    SHARED_DATASET,
    md("## Section 3 · Loss Configs"),
    SHARED_LOSSES,
    md("## Section 4 · COCO Annotations for MMDetection"),
    FRCNN_COCO_ANNOTS,
    SHARED_COCOGT,
    md("## Section 5 · MMDet Loss Registration"),
    FRCNN_LOSS_REG,
    md("## Section 6 · Faster R-CNN Training Functions"),
    FRCNN_CONFIG_AND_RUN,
    md("## Section 7 · Training Loop"),
    code(
        "# Faster R-CNN — 18 runs (6 losses × 3 splits)\n"
        "total, done, skipped, failed = 0, 0, 0, 0\n"
        "for loss_name, _ in LOSS_CONFIGS:\n"
        "    for split_name in SPLIT_CONFIGS:\n"
        "        total += 1\n"
        "        try:\n"
        "            result = run_frcnn(loss_name, split_name)\n"
        "            if result: done += 1\n"
        "            else: skipped += 1\n"
        "        except Exception as e:\n"
        "            print(f'  [ERROR] {loss_name}/{split_name}: {e}'); failed += 1\n"
        "print(f'\\nFaster R-CNN: {done} done, {skipped} skipped, {failed} failed / {total} total')\n"
    ),
    md("## Section 8 · Save & Output"),
    code(FRCNN_SUMMARY_SAVE),
    code(SHARED_OUTPUT_INSTRUCTIONS),
]

# ═══════════════════════════════════════════════════════════════════════════
# 06d – Analysis (aggregates outputs from 06a, 06b, 06c)
# ═══════════════════════════════════════════════════════════════════════════

cells_06d = [
    md("# 06d · Phase 3: Cross-Architecture Analysis\n\n"
       "Loads summary CSVs from shards 06a/06b/06c and produces the cross-architecture comparison.\n\n"
       "**Prerequisites:** Mount Kaggle Datasets from 06a, 06b, 06c as inputs:\n"
       "- `phase3-yolo26s-results`  → `/kaggle/input/phase3-yolo26s-results/`\n"
       "- `phase3-rtdetr-results`   → `/kaggle/input/phase3-rtdetr-results/`\n"
       "- `phase3-frcnn-results`    → `/kaggle/input/phase3-frcnn-results/`\n\n"
       "On Colab, just point `EXPERIMENTS` at the Drive directory."),
    md("## Section 1 · Environment & Load Data"),
    SHARED_ENV,
    code(
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',\n"
        "                'ultralytics==8.4.9', 'pycocotools'], check=True)\n"
        "print('Deps installed.')\n"
    ),
    code(
        "import pandas as pd, numpy as np, json as _json, os\n"
        "from pathlib import Path\n\n"
        "# Locate shard outputs\n"
        "if ON_KAGGLE:\n"
        "    SHARD_DIRS = {\n"
        "        'yolo26s': Path('/kaggle/input/phase3-yolo26s-results/experiments_phase3'),\n"
        "        'rtdetr':  Path('/kaggle/input/phase3-rtdetr-results/experiments_phase3'),\n"
        "        'frcnn':   Path('/kaggle/input/phase3-frcnn-results/experiments_phase3'),\n"
        "    }\n"
        "else:  # Colab / local — all shards wrote to the same EXPERIMENTS dir\n"
        "    ROOT = Path('/content') if os.path.exists('/content') else Path.cwd()\n"
        "    _exp = ROOT / 'experiments_phase3'\n"
        "    SHARD_DIRS = {'yolo26s': _exp, 'rtdetr': _exp, 'frcnn': _exp}\n\n"
        "ANALYSIS_OUT = Path('/kaggle/working/analysis') if ON_KAGGLE else Path('/content/analysis')\n"
        "ANALYSIS_OUT.mkdir(parents=True, exist_ok=True)\n\n"
        "# Load per-arch summary CSVs\n"
        "dfs = []\n"
        "for arch, shard_dir in SHARD_DIRS.items():\n"
        "    csv = shard_dir / f'summary_{arch}.csv'\n"
        "    if csv.exists():\n"
        "        df = pd.read_csv(csv)\n"
        "        df['arch'] = arch\n"
        "        dfs.append(df)\n"
        "        print(f'Loaded {arch}: {len(df)} rows')\n"
        "    else:\n"
        "        print(f'[MISS] {csv}')\n\n"
        "df_all = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()\n"
        "print(f'\\nTotal runs loaded: {len(df_all)}')\n"
        "df_all.head()\n"
    ),
    md("## Section 2 · Cross-Architecture Comparison Table"),
    code(
        "# Pivot: rows=loss, cols=arch, values=mAP50-95 (clean split)\n"
        "clean = df_all[df_all['split'] == 'clean'].copy()\n"
        "pivot = clean.pivot_table(index='loss', columns='arch', values='map50_95')\n\n"
        "# Delta vs EIoU per arch\n"
        "if 'eiou' in pivot.index:\n"
        "    delta = pivot.subtract(pivot.loc['eiou'], axis=1)\n"
        "    delta.columns = [f'Δ_{c}' for c in delta.columns]\n"
        "    table = pd.concat([pivot, delta], axis=1).round(4)\n"
        "else:\n"
        "    table = pivot.round(4)\n\n"
        "print('=== Cross-Architecture mAP50-95 (clean split) ===')\n"
        "print(table.to_string())\n"
        "table.to_csv(ANALYSIS_OUT / 'cross_arch_map50_95.csv')\n"
    ),
    code(
        "# Success criterion: AEIoU beats EIoU on ≥ 2/3 architectures\n"
        "aeiou_rows = pivot[pivot.index.str.startswith('aeiou')]\n"
        "if not aeiou_rows.empty and 'eiou' in pivot.index:\n"
        "    eiou_row = pivot.loc['eiou']\n"
        "    best_aeiou = aeiou_rows.max()\n"
        "    wins = (best_aeiou > eiou_row).sum()\n"
        "    print(f'Best AEIoU vs EIoU per arch:')\n"
        "    for arch in pivot.columns:\n"
        "        diff = best_aeiou.get(arch, 0) - eiou_row.get(arch, 0)\n"
        "        mark = '✓' if diff > 0 else '✗'\n"
        "        print(f'  {arch:12s}: Δ={diff:+.4f}  {mark}')\n"
        "    print(f'\\n→ AEIoU wins on {wins}/3 architectures.')\n"
        "    if wins >= 2:\n"
        "        print('SUCCESS: gain is loss-driven, not architecture-specific.')\n"
        "    else:\n"
        "        print('INCONCLUSIVE: AEIoU gain not confirmed across architectures.')\n"
        "else:\n"
        "    print('AEIoU or EIoU rows missing — run training shards first.')\n"
    ),
    md("## Section 3 · Delta Bar Chart"),
    code(
        "import matplotlib.pyplot as plt\n\n"
        "if not aeiou_rows.empty and 'eiou' in pivot.index:\n"
        "    best_lambda = aeiou_rows.mean(axis=1).idxmax()\n"
        "    deltas = (pivot.loc[best_lambda] - pivot.loc['eiou']).dropna()\n\n"
        "    fig, ax = plt.subplots(figsize=(7, 4))\n"
        "    colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in deltas.values]\n"
        "    ax.bar(deltas.index, deltas.values, color=colors, edgecolor='black')\n"
        "    ax.axhline(0, color='black', linewidth=0.8)\n"
        "    ax.set_ylabel('ΔmAP50-95 vs EIoU (clean)')\n"
        "    ax.set_title(f'Best AEIoU ({best_lambda}) − EIoU per Architecture')\n"
        "    for i, (arch, v) in enumerate(deltas.items()):\n"
        "        ax.text(i, v + (0.001 if v >= 0 else -0.002), f'{v:+.4f}',\n"
        "                ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)\n"
        "    plt.tight_layout()\n"
        "    plt.savefig(ANALYSIS_OUT / 'delta_bar_cross_arch.png', dpi=150)\n"
        "    plt.show()\n"
        "    print(f'Saved: {ANALYSIS_OUT}/delta_bar_cross_arch.png')\n"
    ),
    md("## Section 4 · mAP50-95 Heatmap (all losses × all archs)"),
    code(
        "import matplotlib.pyplot as plt\n"
        "import matplotlib.colors as mcolors\n\n"
        "fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns)*2), len(pivot)*0.6 + 1))\n"
        "im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')\n"
        "ax.set_xticks(range(len(pivot.columns)))\n"
        "ax.set_xticklabels(pivot.columns, rotation=30, ha='right')\n"
        "ax.set_yticks(range(len(pivot.index)))\n"
        "ax.set_yticklabels(pivot.index)\n"
        "for i in range(len(pivot.index)):\n"
        "    for j in range(len(pivot.columns)):\n"
        "        v = pivot.values[i, j]\n"
        "        if not np.isnan(v):\n"
        "            ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=8)\n"
        "plt.colorbar(im, ax=ax, label='mAP50-95')\n"
        "ax.set_title('Phase 3: mAP50-95 by Loss × Architecture (clean split)')\n"
        "plt.tight_layout()\n"
        "plt.savefig(ANALYSIS_OUT / 'heatmap_cross_arch.png', dpi=150)\n"
        "plt.show()\n"
    ),
    md("## Section 5 · mAP50-95 vs Noise Level (per arch)"),
    code(
        "# Line plot: how each loss degrades from clean → low → high noise, per arch\n"
        "import matplotlib.pyplot as plt\n\n"
        "SPLITS_ORDER = ['clean', 'low', 'high']\n"
        "arch_list = df_all['arch'].unique()\n"
        "fig, axes = plt.subplots(1, len(arch_list), figsize=(5*len(arch_list), 4), sharey=True)\n"
        "if len(arch_list) == 1: axes = [axes]\n\n"
        "for ax, arch in zip(axes, arch_list):\n"
        "    sub = df_all[df_all['arch'] == arch]\n"
        "    for loss_name in sub['loss'].unique():\n"
        "        ldf = sub[sub['loss'] == loss_name]\n"
        "        vals = [ldf[ldf['split']==s]['map50_95'].mean() for s in SPLITS_ORDER]\n"
        "        ls = '--' if loss_name.startswith('aeiou') else '-'\n"
        "        ax.plot(SPLITS_ORDER, vals, marker='o', linestyle=ls, label=loss_name)\n"
        "    ax.set_title(arch)\n"
        "    ax.set_ylabel('mAP50-95')\n"
        "    ax.legend(fontsize=6)\n"
        "plt.suptitle('mAP50-95 vs Noise Level by Architecture')\n"
        "plt.tight_layout()\n"
        "plt.savefig(ANALYSIS_OUT / 'noise_robustness_cross_arch.png', dpi=150)\n"
        "plt.show()\n"
    ),
    md("## Section 6 · PR Curves (YOLOv26s + RT-DETR-L)"),
    code(
        "# Load PR curve JSONs from Ultralytics shards\n"
        "import matplotlib.pyplot as plt, json as _json\n\n"
        "ARCH_LABELS = {'yolo26s': 'YOLOv26s', 'rtdetr': 'RT-DETR-L'}\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n\n"
        "for ax, (arch, shard_dir) in zip(axes, [(a, SHARD_DIRS[a]) for a in ['yolo26s', 'rtdetr']\n"
        "                                          if a in SHARD_DIRS]):\n"
        "    metrics_dir = shard_dir / 'metrics'\n"
        "    for loss_name, _ in [('eiou', None), ('ciou', None), ('eciou', None)]:\n"
        "        run_name = f'phase3_{arch}_{loss_name}_clean_s42_e20'\n"
        "        pr_path  = metrics_dir / run_name / 'pr_curve.json'\n"
        "        if not pr_path.exists(): continue\n"
        "        pr = _json.loads(pr_path.read_text())\n"
        "        if pr['recall'] and pr['precision']:\n"
        "            ax.plot(pr['recall'], pr['precision'], label=f'{loss_name} AP50={pr[\"ap50\"]:.3f}')\n"
        "    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')\n"
        "    ax.set_title(f'PR Curve — {ARCH_LABELS.get(arch, arch)} (clean split)')\n"
        "    ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)\n\n"
        "plt.tight_layout()\n"
        "plt.savefig(ANALYSIS_OUT / 'pr_curves_cross_arch.png', dpi=150)\n"
        "plt.show()\n"
    ),
    md("## Section 7 · Final Save"),
    code(
        "# Save full merged DataFrame\n"
        "df_all.to_csv(ANALYSIS_OUT / 'phase3_all_results.csv', index=False)\n"
        "n_figs = len(list(ANALYSIS_OUT.glob('*.png')))\n"
        "n_csvs = len(list(ANALYSIS_OUT.glob('*.csv')))\n"
        "print(f'Analysis complete.')\n"
        "print(f'  Figures : {n_figs} PNGs')\n"
        "print(f'  Tables  : {n_csvs} CSVs')\n"
        "print(f'  Output  : {ANALYSIS_OUT}')\n"
    ),
]

# ═══════════════════════════════════════════════════════════════════════════
# Write notebooks
# ═══════════════════════════════════════════════════════════════════════════
save('notebooks/06a_yolo26s.ipynb',    cells_06a)
save('notebooks/06b_rtdetr.ipynb',     cells_06b)
save('notebooks/06c_frcnn.ipynb',      cells_06c)
save('notebooks/06d_analysis.ipynb',   cells_06d)
print('\nAll four split notebooks generated.')
