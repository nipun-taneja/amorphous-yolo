import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('notebooks/03_kvasir_eiou_vs_aeiou.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

def code_cell(source):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': source}

def markdown_cell(source):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': source}

# ---------------------------------------------------------------------------
# Section 15 header
# ---------------------------------------------------------------------------
nb['cells'].append(markdown_cell(
    "## Section 15 · PR Curves, COCO AP Suite & AP vs IoU Threshold\n\n"
    "All plots load from persisted JSON files in `experiments_kvasir/metrics/` —\n"
    "**no retraining or re-validation required**. Run Cells A–E (Section 14) first.\n\n"
    "| Cell | Plot | Source |\n"
    "|---|---|---|\n"
    "| F | PR curves — all 18 losses × 3 splits | `pr_curve.json` |\n"
    "| G | COCO AP suite heatmap (AP50/75/APs/APm/APl) | `metrics_all_losses.json` |\n"
    "| H | AP vs IoU threshold — all losses + focused comparison | `pr_curve.json` |"
))

# ---------------------------------------------------------------------------
# Cell F — PR curves
# ---------------------------------------------------------------------------
cell_f = '''\
# --- Cell F: PR curves — all 18 losses x 3 splits (loaded from pr_curve.json)
# Three subplots (one per split) each showing all 18 loss PR curves.
# Plus a focused inset: top-5 losses by AP50 on the clean split.
# Figures saved to experiments_kvasir/analysis/
import json as _json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

_all_loss_keys = BASELINE_LOSS_NAMES + [f"aeiou_r{_fmt_r(r)}" for r in AEIOU_RIGIDITIES]
_seed = SEEDS[0]
_splits = ["clean", "low", "high"]
_split_titles = {"clean": "Clean split", "low": "Low-noise split", "high": "High-noise split"}

# ── Load all PR data ──────────────────────────────────────────────────────────
pr_store = {}   # (loss, split) -> dict with precision/recall/ap50
for _loss in _all_loss_keys:
    for _split in _splits:
        _run = f"kvasir_yolo26n_{_loss}_{_split}_s{_seed}_e{EPOCHS}"
        _fpath = METRICS_DIR / _run / "pr_curve.json"
        if _fpath.exists():
            pr_store[(_loss, _split)] = _json.loads(_fpath.read_text())

print(f"Loaded PR data for {len(pr_store)} runs")

# ── Fig F1: All 18 losses x 3 splits (3-panel) ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig.suptitle("Precision-Recall Curves — all losses, all splits", fontsize=14, fontweight="bold")

for ax, _split in zip(axes, _splits):
    for _loss in _all_loss_keys:
        d = pr_store.get((_loss, _split))
        if d is None or not d.get("precision") or not d.get("recall"):
            continue
        prec = np.array(d["precision"])
        rec  = np.array(d["recall"])
        # Sort by recall for clean plotting
        order = np.argsort(rec)
        color = PALETTE.get(_loss, "#aaaaaa")
        lw = 2.5 if "aeiou" in _loss else 1.2
        alpha = 0.9 if "aeiou" in _loss else 0.55
        label = LOSS_LABELS.get(_loss, _loss)
        ap = d.get("ap50", 0.0)
        ax.plot(rec[order], prec[order], color=color, lw=lw, alpha=alpha,
                label=f"{label} ({ap:.3f})")

    ax.set_title(_split_titles[_split], fontsize=12)
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)

axes[0].set_ylabel("Precision", fontsize=10)

# Shared legend — two columns, outside right panel
handles = []
for _loss in _all_loss_keys:
    d = pr_store.get((_loss, "clean"))
    ap = d["ap50"] if d else 0.0
    handles.append(mlines.Line2D([], [], color=PALETTE.get(_loss, "#aaa"),
                                  lw=2, label=f"{LOSS_LABELS.get(_loss, _loss)} ({ap:.3f})"))
axes[2].legend(handles=handles, fontsize=7.5, loc="lower left",
               ncol=2, framealpha=0.85, title="Loss (AP50 clean)")

plt.tight_layout()
_save = ANALYSIS_DIR / "F1_pr_curves_all_splits.png"
plt.savefig(_save, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved -> {_save}")

# ── Fig F2: Focused — top-5 losses by AP50 on clean split ────────────────────
_ap50_clean = {
    _loss: pr_store[(_loss, "clean")]["ap50"]
    for _loss in _all_loss_keys
    if (_loss, "clean") in pr_store and pr_store[(_loss, "clean")].get("ap50") is not None
}
_top5 = sorted(_ap50_clean, key=_ap50_clean.get, reverse=True)[:5]

fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig2.suptitle("PR Curves — top-5 losses by AP50 (clean split)", fontsize=13, fontweight="bold")

for ax, _split in zip(axes2, _splits):
    for _loss in _top5:
        d = pr_store.get((_loss, _split))
        if d is None or not d.get("precision"):
            continue
        prec = np.array(d["precision"])
        rec  = np.array(d["recall"])
        order = np.argsort(rec)
        label = f"{LOSS_LABELS.get(_loss, _loss)} (AP50={d.get('ap50', 0):.3f})"
        ax.plot(rec[order], prec[order], color=PALETTE.get(_loss, "#888"),
                lw=2.5, label=label)
    ax.set_title(_split_titles[_split], fontsize=11)
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.85)

axes2[0].set_ylabel("Precision", fontsize=10)
plt.tight_layout()
_save2 = ANALYSIS_DIR / "F2_pr_curves_top5.png"
plt.savefig(_save2, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved -> {_save2}")
print(f"Top-5 losses by AP50 (clean): {[LOSS_LABELS.get(l, l) for l in _top5]}")
'''
nb['cells'].append(code_cell(cell_f))

# ---------------------------------------------------------------------------
# Cell G — COCO AP suite heatmap
# ---------------------------------------------------------------------------
cell_g = '''\
# --- Cell G: COCO AP suite heatmap — AP50, AP75, mAP50-95, APs, APm, APl
# Rows = losses (sorted by mAP50-95 on clean split), columns = metric x split.
# Loads from metrics_all_losses.json — no model inference needed.
import json as _json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_unified = EXPERIMENTS / "metrics_all_losses.json"
if not _unified.exists():
    print("[SKIP] metrics_all_losses.json not found. Run Cell D first.")
else:
    _all_m = _json.loads(_unified.read_text())
    _all_loss_keys = BASELINE_LOSS_NAMES + [f"aeiou_r{_fmt_r(r)}" for r in AEIOU_RIGIDITIES]
    _seed = SEEDS[0]
    _splits = ["clean", "low", "high"]
    _metrics = ["map50_95", "map50", "map75", "APs", "APm", "APl"]
    _metric_labels = {
        "map50_95": "mAP\n50-95", "map50": "mAP\n50", "map75": "mAP\n75",
        "APs": "AP\nsmall", "APm": "AP\nmedium", "APl": "AP\nlarge"
    }

    # Build wide DataFrame: rows=loss, columns=(metric, split)
    rows = {}
    for _loss in _all_loss_keys:
        rows[_loss] = {}
        for _split in _splits:
            _run = f"kvasir_yolo26n_{_loss}_{_split}_s{_seed}_e{EPOCHS}"
            m = _all_m.get(_run, {})
            for _met in _metrics:
                rows[_loss][(_met, _split)] = m.get(_met)

    df_wide = pd.DataFrame(rows).T
    df_wide.columns = pd.MultiIndex.from_tuples(df_wide.columns)

    # ── Fig G1: One heatmap per metric (6 panels) ─────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("COCO AP Suite — all losses x all splits", fontsize=14, fontweight="bold")

    # Sort rows by mAP50-95 on clean split
    _sort_key = [rows[l].get(("map50_95", "clean")) or 0 for l in _all_loss_keys]
    _sorted_losses = [l for _, l in sorted(zip(_sort_key, _all_loss_keys), reverse=True)]
    _display_labels = [LOSS_LABELS.get(l, l) for l in _sorted_losses]

    for ax, _met in zip(axes.flat, _metrics):
        data = []
        for _loss in _sorted_losses:
            data.append([rows[_loss].get((_met, _sp)) for _sp in _splits])
        arr = np.array(data, dtype=float)
        # Replace -1 (pycocotools "no detections") with NaN
        arr[arr < 0] = np.nan
        _vmax = np.nanmax(arr) if not np.all(np.isnan(arr)) else 1.0
        sns.heatmap(
            arr, ax=ax,
            xticklabels=_splits,
            yticklabels=_display_labels,
            annot=True, fmt=".3f", annot_kws={"size": 7},
            cmap="YlGn", vmin=0, vmax=_vmax,
            linewidths=0.3, linecolor="#cccccc",
            cbar_kws={"shrink": 0.7},
        )
        ax.set_title(_metric_labels[_met], fontsize=11)
        ax.tick_params(axis='y', labelsize=7)
        ax.tick_params(axis='x', labelsize=9)

    plt.tight_layout()
    _save = ANALYSIS_DIR / "G1_coco_ap_suite_heatmap.png"
    plt.savefig(_save, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {_save}")

    # ── Fig G2: Clean-split bar chart — all 6 metrics side by side ─────────────
    fig2, ax2 = plt.subplots(figsize=(16, 6))
    _n = len(_sorted_losses)
    _x = np.arange(_n)
    _width = 0.13
    _colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4"]

    for j, _met in enumerate(_metrics):
        vals = [rows[l].get((_met, "clean")) for l in _sorted_losses]
        vals = [v if (v is not None and v >= 0) else 0.0 for v in vals]
        ax2.bar(_x + j * _width, vals, _width, label=_metric_labels[_met].replace("\n", " "),
                color=_colors[j], alpha=0.85)

    ax2.set_xticks(_x + _width * 2.5)
    ax2.set_xticklabels(_display_labels, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("AP", fontsize=11)
    ax2.set_title("COCO AP Suite — clean split, all losses", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save2 = ANALYSIS_DIR / "G2_coco_ap_bar_clean.png"
    plt.savefig(_save2, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {_save2}")
'''
nb['cells'].append(code_cell(cell_g))

# ---------------------------------------------------------------------------
# Cell H — AP vs IoU threshold
# ---------------------------------------------------------------------------
cell_h = '''\
# --- Cell H: AP vs IoU threshold curves (0.50 -> 0.95, step 0.05)
# ap_per_iou_threshold[10] is already in pr_curve.json — no re-validation needed.
# Plots:
#   H1: All 18 losses on clean split (line per loss)
#   H2: EIoU vs best AEIoU on all 3 splits (2x1 grid showing degradation)
import json as _json
import numpy as np
import matplotlib.pyplot as plt

_all_loss_keys = BASELINE_LOSS_NAMES + [f"aeiou_r{_fmt_r(r)}" for r in AEIOU_RIGIDITIES]
_seed = SEEDS[0]
_splits = ["clean", "low", "high"]
_iou_thresholds = np.round(np.arange(0.50, 1.00, 0.05), 2)

# ── Load ap_per_iou_threshold for all runs ────────────────────────────────────
ap_thresh_store = {}
for _loss in _all_loss_keys:
    for _split in _splits:
        _run = f"kvasir_yolo26n_{_loss}_{_split}_s{_seed}_e{EPOCHS}"
        _fpath = METRICS_DIR / _run / "pr_curve.json"
        if _fpath.exists():
            d = _json.loads(_fpath.read_text())
            if d.get("ap_per_iou_threshold"):
                ap_thresh_store[(_loss, _split)] = np.array(d["ap_per_iou_threshold"])

print(f"Loaded AP-vs-threshold data for {len(ap_thresh_store)} runs")

# ── Fig H1: All 18 losses on clean split ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

for _loss in _all_loss_keys:
    d = ap_thresh_store.get((_loss, "clean"))
    if d is None:
        continue
    color = PALETTE.get(_loss, "#aaa")
    lw = 2.5 if "aeiou" in _loss else 1.2
    alpha = 0.9 if "aeiou" in _loss else 0.55
    label = LOSS_LABELS.get(_loss, _loss)
    ax.plot(_iou_thresholds, d, color=color, lw=lw, alpha=alpha,
            marker="o", markersize=3, label=label)

ax.set_xlabel("IoU threshold", fontsize=11)
ax.set_ylabel("Average Precision", fontsize=11)
ax.set_title("AP vs IoU Threshold — all losses, clean split", fontsize=13, fontweight="bold")
ax.set_xticks(_iou_thresholds)
ax.set_xticklabels([f"{t:.2f}" for t in _iou_thresholds], fontsize=8)
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, ncol=3, loc="upper right", framealpha=0.85)
plt.tight_layout()
_save = ANALYSIS_DIR / "H1_ap_vs_iou_all_losses_clean.png"
plt.savefig(_save, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved -> {_save}")

# ── Fig H2: EIoU vs best AEIoU — all 3 splits (robustness view) ──────────────
# Identify best AEIoU by AP50 on clean split
_aeiou_ap50 = {
    l: ap_thresh_store[(_l := l, "clean")][0]
    for l in _all_loss_keys if "aeiou" in l and (_l, "clean") in ap_thresh_store
}
_best_aeiou = max(_aeiou_ap50, key=_aeiou_ap50.get) if _aeiou_ap50 else "aeiou_r0p3"
_focus_losses = ["eiou", "eciou", _best_aeiou]
_focus_colors = {"eiou": "#E63946", "eciou": "#9B2226", _best_aeiou: "#00B4D8"}
_focus_labels = {l: LOSS_LABELS.get(l, l) for l in _focus_losses}

_split_ls = {"clean": "-", "low": "--", "high": ":"}
_split_colors_map = {"clean": "#2E7D32", "low": "#F57F17", "high": "#B71C1C"}

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig2.suptitle(
    f"AP vs IoU Threshold — EIoU vs ECIoU vs {_focus_labels[_best_aeiou]}, all splits",
    fontsize=13, fontweight="bold"
)

for ax2, _loss in zip(axes2, ["eiou", _best_aeiou]):
    for _split in _splits:
        d = ap_thresh_store.get((_loss, _split))
        if d is None:
            continue
        _lbl = f"{LOSS_LABELS.get(_loss, _loss)} — {_split}"
        ax2.plot(_iou_thresholds, d,
                 color=_split_colors_map[_split],
                 ls=_split_ls[_split], lw=2.2,
                 marker="o", markersize=4,
                 label=_lbl)
    ax2.set_title(LOSS_LABELS.get(_loss, _loss), fontsize=11)
    ax2.set_xlabel("IoU threshold", fontsize=10)
    ax2.set_xticks(_iou_thresholds)
    ax2.set_xticklabels([f"{t:.2f}" for t in _iou_thresholds], fontsize=7, rotation=45)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9, framealpha=0.85)

axes2[0].set_ylabel("Average Precision", fontsize=10)
plt.tight_layout()
_save2 = ANALYSIS_DIR / "H2_ap_vs_iou_eiou_vs_aeiou_splits.png"
plt.savefig(_save2, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved -> {_save2}")
print(f"Best AEIoU used: {_best_aeiou} (AP50 clean = {_aeiou_ap50.get(_best_aeiou, 0):.4f})")

# ── Fig H3: Area-under-curve summary (AUC of AP-vs-threshold = mAP50-95) ─────
# This is exactly mAP50-95 — verify our JSON values match the COCO suite values.
fig3, ax3 = plt.subplots(figsize=(14, 5))
_aucs = []
for _loss in _all_loss_keys:
    d = ap_thresh_store.get((_loss, "clean"))
    auc = float(np.mean(d)) if d is not None else 0.0
    _aucs.append(auc)

_colors_bar = [PALETTE.get(l, "#aaa") for l in _all_loss_keys]
_x = np.arange(len(_all_loss_keys))
ax3.bar(_x, _aucs, color=_colors_bar, alpha=0.85, edgecolor="white", linewidth=0.5)
ax3.set_xticks(_x)
ax3.set_xticklabels(
    [LOSS_LABELS.get(l, l) for l in _all_loss_keys],
    rotation=45, ha="right", fontsize=8
)
ax3.set_ylabel("Mean AP (= mAP50-95)", fontsize=10)
ax3.set_title("mAP50-95 derived from AP-per-threshold — clean split (cross-check)", fontsize=12)
ax3.grid(axis="y", alpha=0.3)
plt.tight_layout()
_save3 = ANALYSIS_DIR / "H3_map5095_crosscheck.png"
plt.savefig(_save3, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved -> {_save3}")
'''
nb['cells'].append(code_cell(cell_h))

# ---------------------------------------------------------------------------
# Drive sync for new figures
# ---------------------------------------------------------------------------
cell_sync = '''\
# --- Sync Section 15 figures to Drive
import shutil

if DRIVE_AVAILABLE:
    try:
        shutil.copytree(str(ANALYSIS_DIR), str(DRIVE_EXPERIMENTS / "analysis"),
                        dirs_exist_ok=True)
        n = len(list(ANALYSIS_DIR.glob("*.png")))
        print(f"Analysis dir synced to Drive ({n} PNGs).")
    except Exception as e:
        print(f"Drive sync failed: {e}")
else:
    n = len(list(ANALYSIS_DIR.glob("*.png")))
    print(f"Section 15 complete. {n} figures in {ANALYSIS_DIR}")
    print("On Kaggle: download from Output tab after run completes.")
'''
nb['cells'].append(code_cell(cell_sync))

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
with open('notebooks/03_kvasir_eiou_vs_aeiou.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Saved: 03_kvasir_eiou_vs_aeiou.ipynb  ({len(nb['cells'])} cells)")
