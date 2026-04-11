#!/usr/bin/env python3
"""Part 3 of the generator — appends analysis cells to the cells list.
This file is imported by gen_03_run.py which combines all parts.
"""

def add_cells_part3(cells):
    def md(source):
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    def code(source):
        return {"cell_type": "code", "execution_count": None,
                "metadata": {}, "outputs": [], "source": source}

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 10 — Summary Table & Pivot
    # ─────────────────────────────────────────────────────────────────────────
    cells.append(md(
"""## Section 10 · Summary Table

The master summary table: final-epoch mAP50 and mAP50-95 for every loss x split,
aggregated across seeds (mean +/- std when multiple seeds are present).
This is the central table for the paper — all analyses derive from it.
"""
    ))

    cells.append(code(
"""# --- Build master pivot table with mean +/- std across seeds
import pandas as pd
import numpy as np

MAP50_COL  = "metrics/mAP50(B)"
MAP95_COL  = "metrics/mAP50-95(B)"

# Take the last epoch per run
df_final = (
    df_all.sort_values("epoch")
          .groupby("run_name")
          .last()
          .reset_index()
)

# Aggregate across seeds: mean and std per (loss, split)
agg_rows = []
for loss_name in ALL_LOSS_KEYS:
    for split in ["clean", "low", "high"]:
        sub = df_final[(df_final["loss"] == loss_name) & (df_final["split"] == split)]
        if sub.empty:
            continue
        row = {
            "loss": loss_name,
            "split": split,
            "label": LOSS_LABELS.get(loss_name, loss_name),
        }
        for col, tag in [(MAP95_COL, "map95"), (MAP50_COL, "map50")]:
            if col in sub.columns:
                row[f"{tag}_mean"] = sub[col].mean()
                row[f"{tag}_std"]  = sub[col].std() if len(sub) > 1 else 0.0
        agg_rows.append(row)

df_agg = pd.DataFrame(agg_rows)

# Pivot for display
pivot95 = df_agg.pivot_table(index="loss", columns="split", values="map95_mean")
pivot95 = pivot95[["clean", "low", "high"]]
pivot95["robust_ratio"] = pivot95["high"] / pivot95["clean"]
pivot95 = pivot95.sort_values("clean", ascending=False)

pivot50 = df_agg.pivot_table(index="loss", columns="split", values="map50_mean")
pivot50 = pivot50[["clean", "low", "high"]]

# Save
pivot95.to_csv(ANALYSIS_DIR / "summary_map95.csv")
pivot50.to_csv(ANALYSIS_DIR / "summary_map50.csv")

print("=== mAP50-95 (localisation quality) ===")
print(pivot95.round(4).to_string())
print(f"\\nBest clean mAP50-95: {pivot95['clean'].idxmax()} = {pivot95['clean'].max():.4f}")
print(f"Best robust ratio:   {pivot95['robust_ratio'].idxmax()} = {pivot95['robust_ratio'].max():.4f}")

print("\\n=== mAP50 (detection quality) ===")
print(pivot50.round(4).to_string())
print(f"\\nBest clean mAP50: {pivot50['clean'].idxmax()} = {pivot50['clean'].max():.4f}")
"""
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 11 — Core Analysis Figures
    # ─────────────────────────────────────────────────────────────────────────
    cells.append(md(
"""## Section 11 · Core Analysis (Research-Level Figures)

15+ figures providing multi-angle evidence for the AEIoU hypothesis.
Each figure is saved to `experiments_kvasir/analysis/` as a high-DPI PNG.

| Figure | Key Question |
|---|---|
| Fig 2: mAP95 bar chart | Which loss achieves highest localisation quality? |
| Fig 2b: mAP50 bar chart | Which loss achieves best detection rate? |
| Fig 3: lambda-vs-mAP curve | Where in the lambda space does AEIoU peak? |
| Fig 4: lambda heatmap | Is optimal lambda stable across noise splits? |
| Fig 5: Learning curves | How do losses compare during training? |
| Fig 6: Convergence speed | Which loss reaches 90% mAP fastest? |
| Fig 7: Noise robustness | Which loss degrades least under annotation noise? |
| Fig 8: PR curves | Detection-recall trade-offs |
| Fig 9: AP@thresholds | Are gains consistent across IoU thresholds? |
| Fig 10: Training stability | Which loss produces smoothest training? |
| Fig 11: Box calibration | Do predicted box dimensions match GT? |
| Fig 12: Statistical tests | Are differences statistically significant? |
| Fig 13: Cross-dataset | Does optimal lambda match notebook 02 (DUO)? |
| Fig 14: Runtime overhead | Does AEIoU add computation cost? |
| Fig 15: IoU distribution | Per-prediction box quality |
| Fig 16: Size-stratified AP | Where does AEIoU help most? |
| Fig 17: Failure cases | Where does EIoU beat AEIoU? |
"""
    ))

    # --- Fig 2: mAP50-95 bar chart (all losses)
    cells.append(code(
"""# --- Fig 2: Final mAP50-95 bar chart — ALL losses grouped by split
import matplotlib.pyplot as plt
import numpy as np

splits = ["clean", "low", "high"]
split_colors = {"clean": "#4CAF50", "low": "#FFC107", "high": "#F44336"}
x_labels = BASELINE_LOSS_NAMES + [f"aeiou_r{_fmt_r(r)}" for r in AEIOU_RIGIDITIES]
display_labels = [LOSS_LABELS.get(l, l) for l in x_labels]
x_pos = np.arange(len(x_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(18, 6))
for i, split in enumerate(splits):
    vals = [pivot95.loc[l, split] if l in pivot95.index else 0 for l in x_labels]
    ax.bar(x_pos + i*width, vals, width=width,
           color=split_colors[split], alpha=0.85, label=split)

# Vertical line separating baselines from AEIoU
ax.axvline(x=len(BASELINE_LOSS_NAMES) - 0.5, color="black", linestyle=":", lw=1.5, alpha=0.4)
ax.text(len(BASELINE_LOSS_NAMES) - 0.3, ax.get_ylim()[1]*0.98, "baselines | AEIoU",
        fontsize=8, rotation=90, va="top")

ax.set_xticks(x_pos + width)
ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("mAP@[.5:.95]", fontsize=11)
ax.set_title("Final mAP50-95: All Losses Across 3 Noise Splits", fontsize=13, fontweight="bold")
ax.legend(title="Split", fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save_path = ANALYSIS_DIR / "02_final_map95_comparison.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved -> {save_path}")
"""
    ))

    # --- Fig 2b: mAP50 bar chart
    cells.append(code(
"""# --- Fig 2b: Final mAP50 bar chart — clinical detection metric
import matplotlib.pyplot as plt
import numpy as np

x_labels = BASELINE_LOSS_NAMES + [f"aeiou_r{_fmt_r(r)}" for r in AEIOU_RIGIDITIES]
display_labels = [LOSS_LABELS.get(l, l) for l in x_labels]
x_pos = np.arange(len(x_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(18, 6))
for i, split in enumerate(["clean", "low", "high"]):
    vals = [pivot50.loc[l, split] if l in pivot50.index else 0 for l in x_labels]
    color = {"clean": "#4CAF50", "low": "#FFC107", "high": "#F44336"}[split]
    ax.bar(x_pos + i*width, vals, width=width, color=color, alpha=0.85, label=split)

ax.axvline(x=len(BASELINE_LOSS_NAMES) - 0.5, color="black", linestyle=":", lw=1.5, alpha=0.4)
ax.set_xticks(x_pos + width)
ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("mAP@0.5 (detection)", fontsize=11)
ax.set_title("Final mAP50: All Losses — Detection Quality (Clinical Metric)",
             fontsize=13, fontweight="bold")
ax.legend(title="Split", fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save_path = ANALYSIS_DIR / "02b_final_map50_comparison.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved -> {save_path}")
"""
    ))

    cells.append(md(
"""**Figures 2 & 2b: mAP50-95 vs mAP50**

mAP50-95 rewards tight, well-sized boxes (localisation). mAP50 rewards finding the
polyp at all (detection). For clinical use, mAP50 matters most — a gastroenterologist
needs to *see* the polyp. But for loss function research, mAP50-95 is the engineering
metric that isolates box quality from detection sensitivity.

**What to look for:** If AEIoU improves mAP50-95 but not mAP50, the gain is purely
in box tightness. If it improves both, the gain extends to detection — a stronger result.
"""
    ))

    # --- Fig 3: lambda-vs-mAP line plot (the "money shot")
    cells.append(code(
"""# --- Fig 3: lambda-vs-mAP continuous curve with baseline reference lines
# This is the key figure — shows exactly where in lambda space AEIoU peaks
# and whether it exceeds ALL baselines, not just EIoU.
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, metric_col, title_suffix in [
    (axes[0], "map95_mean", "mAP50-95 (localisation)"),
    (axes[1], "map50_mean", "mAP50 (detection)"),
]:
    for split, ls, marker in [("clean", "-", "o"), ("low", "--", "s"), ("high", ":", "^")]:
        # AEIoU curve: lambda vs mAP
        lambdas = []
        maps = []
        for r in AEIOU_RIGIDITIES:
            loss_name = f"aeiou_r{_fmt_r(r)}"
            row = df_agg[(df_agg["loss"] == loss_name) & (df_agg["split"] == split)]
            if not row.empty:
                lambdas.append(r)
                maps.append(row[metric_col].values[0])
        if lambdas:
            ax.plot(lambdas, maps, ls + marker, color="#00B4D8", lw=2,
                    markersize=5, label=f"AEIoU ({split})" if split == "clean" else f"_({split})")

        # Horizontal reference lines for each baseline
        for base_name in BASELINE_LOSS_NAMES:
            row = df_agg[(df_agg["loss"] == base_name) & (df_agg["split"] == split)]
            if not row.empty:
                val = row[metric_col].values[0]
                color = PALETTE.get(base_name, "#888")
                if split == "clean":
                    ax.axhline(y=val, color=color, linestyle="-.", lw=1.2, alpha=0.7,
                               label=LOSS_LABELS.get(base_name, base_name))

    ax.set_xlabel("AEIoU rigidity (lambda)", fontsize=11)
    ax.set_ylabel(title_suffix, fontsize=11)
    ax.set_title(f"AEIoU lambda sweep vs baselines — {title_suffix}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_xlim(0.05, 1.05)

plt.tight_layout()
save_path = ANALYSIS_DIR / "03_lambda_vs_map.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved -> {save_path}")
"""
    ))

    cells.append(md(
"""**Figure 3 · Lambda-vs-mAP Curve (The Key Plot)**

*x-axis:* AEIoU lambda (0.1 to 1.0) · *y-axis:* mAP · *Horizontal lines:* baselines
*Solid/dashed/dotted:* clean / low / high splits

**This is the most important figure.** If the AEIoU curve exceeds ALL baseline
reference lines at some lambda, AEIoU is the best loss for polyp detection.
The peak lambda is the recommended setting. If it matches notebook 02 (DUO),
that is strong evidence of a domain-agnostic optimal lambda.
"""
    ))

    # --- Fig 4: lambda sensitivity heatmap
    cells.append(code(
"""# --- Fig 4: Lambda sensitivity heatmap (mAP50-95 and mAP50 side by side)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, metric, title in [
    (axes[0], "map95_mean", "mAP50-95"),
    (axes[1], "map50_mean", "mAP50"),
]:
    heat_data = []
    for r in AEIOU_RIGIDITIES:
        loss_name = f"aeiou_r{_fmt_r(r)}"
        row_data = {"lambda": r}
        for split in ["clean", "low", "high"]:
            sub = df_agg[(df_agg["loss"] == loss_name) & (df_agg["split"] == split)]
            row_data[split] = sub[metric].values[0] if not sub.empty else float("nan")
        heat_data.append(row_data)

    heat_df = pd.DataFrame(heat_data).set_index("lambda")
    sns.heatmap(heat_df.T, annot=True, fmt=".4f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": title})
    ax.set_title(f"AEIoU lambda sensitivity — {title}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Rigidity (lambda)", fontsize=10)
    ax.set_ylabel("Split", fontsize=10)

plt.tight_layout()
save_path = ANALYSIS_DIR / "04_lambda_heatmap.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()

# Report optimal lambda per split
for split in ["clean", "low", "high"]:
    sub = df_agg[df_agg["split"] == split]
    aeiou_sub = sub[sub["loss"].str.startswith("aeiou")]
    if not aeiou_sub.empty:
        best_row = aeiou_sub.loc[aeiou_sub["map95_mean"].idxmax()]
        print(f"  Best lambda for {split}: {best_row['loss']}  mAP95={best_row['map95_mean']:.4f}")
print(f"Saved -> {save_path}")
"""
    ))

    # --- Fig 5: Learning curves (all losses, clean split)
    cells.append(code(
"""# --- Fig 5: Learning curves — train loss + val mAP for ALL losses (clean split)
import matplotlib.pyplot as plt

MAP95_COL = "metrics/mAP50-95(B)"
LOSS_COL  = "train/box_loss"

# Show baselines + select AEIoU (0.1, 0.3, 0.5, 1.0) for readability
vis_losses = BASELINE_LOSS_NAMES + ["aeiou_r0p1", "aeiou_r0p3", "aeiou_r0p5", "aeiou_r1p0"]

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle("Learning curves — clean split (all losses)", fontsize=13, fontweight="bold")

for loss_name in vis_losses:
    color = PALETTE.get(loss_name, "#888")
    label = LOSS_LABELS.get(loss_name, loss_name)

    # Average across seeds for this loss on clean split
    matching = df_all[(df_all["loss"] == loss_name) & (df_all["split"] == "clean")]
    if matching.empty:
        continue
    avg = matching.groupby("epoch").mean(numeric_only=True).reset_index()

    lw = 2.5 if loss_name in ["eiou", "eciou"] or "aeiou" in loss_name else 1.2
    ls = "-" if loss_name in BASELINE_LOSS_NAMES else "--"

    if LOSS_COL in avg.columns:
        axes[0].plot(avg["epoch"], avg[LOSS_COL], color=color, lw=lw, ls=ls, label=label)
    if MAP95_COL in avg.columns:
        axes[1].plot(avg["epoch"], avg[MAP95_COL], color=color, lw=lw, ls=ls, label=label)

axes[0].set_ylabel("Training box loss"); axes[0].legend(fontsize=7, ncol=3); axes[0].grid(alpha=0.3)
axes[1].set_ylabel("Val mAP50-95"); axes[1].set_xlabel("Epoch")
axes[1].legend(fontsize=7, ncol=3); axes[1].grid(alpha=0.3)

plt.tight_layout()
save_path = ANALYSIS_DIR / "05_learning_curves.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved -> {save_path}")
"""
    ))

    # --- Fig 6: Convergence speed
    cells.append(code(
"""# --- Fig 6: Convergence speed — epoch to reach 90% of final mAP
import matplotlib.pyplot as plt
import numpy as np

MAP95_COL = "metrics/mAP50-95(B)"
threshold = 0.90

all_losses = BASELINE_LOSS_NAMES + [f"aeiou_r{_fmt_r(r)}" for r in AEIOU_RIGIDITIES]
conv_data = {split: [] for split in ["clean", "low", "high"]}

for loss_name in all_losses:
    for split in ["clean", "low", "high"]:
        sub = df_all[(df_all["loss"] == loss_name) & (df_all["split"] == split)]
        if sub.empty or MAP95_COL not in sub.columns:
            conv_data[split].append(EPOCHS)
            continue
        avg = sub.groupby("epoch")[MAP95_COL].mean().reset_index().sort_values("epoch")
        final_map = avg[MAP95_COL].iloc[-1]
        target = threshold * final_map
        reached = avg[avg[MAP95_COL] >= target]["epoch"]
        conv_epoch = int(reached.iloc[0]) if len(reached) else EPOCHS
        conv_data[split].append(conv_epoch)

x_pos = np.arange(len(all_losses))
display_labels = [LOSS_LABELS.get(l, l) for l in all_losses]
width = 0.25
fig, ax = plt.subplots(figsize=(16, 5))
for i, (split, color) in enumerate(zip(["clean","low","high"],
                                        ["#4CAF50","#FFC107","#F44336"])):
    ax.bar(x_pos + i*width, conv_data[split], width=width, color=color, alpha=0.85, label=split)

ax.axvline(x=len(BASELINE_LOSS_NAMES) - 0.5, color="black", linestyle=":", lw=1.5, alpha=0.4)
ax.set_xticks(x_pos + width)
ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel(f"Epoch to reach {threshold*100:.0f}% of final mAP", fontsize=11)
ax.set_title("Convergence speed: epochs to 90% final mAP", fontsize=13, fontweight="bold")
ax.legend(title="Split")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save_path = ANALYSIS_DIR / "06_convergence_speed.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved -> {save_path}")
"""
    ))

    # --- Fig 7: Noise robustness gap
    cells.append(code(
"""# --- Fig 7: Noise robustness gap (mAP_clean - mAP_high) — smaller = more robust
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

all_losses = BASELINE_LOSS_NAMES + [f"aeiou_r{_fmt_r(r)}" for r in AEIOU_RIGIDITIES]
gaps = []
for loss_name in all_losses:
    clean_map = pivot95.loc[loss_name, "clean"] if loss_name in pivot95.index else 0
    high_map  = pivot95.loc[loss_name, "high"]  if loss_name in pivot95.index else 0
    gaps.append(clean_map - high_map)

display_labels = [LOSS_LABELS.get(l, l) for l in all_losses]

# Color: baselines in palette, AEIoU in blue gradient
bar_colors = [PALETTE.get(l, "#888") for l in all_losses]

fig, ax = plt.subplots(figsize=(16, 5))
bars = ax.bar(display_labels, gaps, color=bar_colors, edgecolor="white", lw=0.5)

# Mark EIoU reference line
eiou_idx = all_losses.index("eiou") if "eiou" in all_losses else 0
ax.axhline(y=gaps[eiou_idx], color="#E63946", linestyle="--", lw=1.5,
           label=f"EIoU gap = {gaps[eiou_idx]:.4f}")

ax.set_ylabel("mAP gap (clean - high) — smaller = more robust", fontsize=11)
ax.set_title("Noise Robustness Gap — All Losses", fontsize=13, fontweight="bold")
ax.tick_params(axis="x", rotation=45)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
for bar, gap in zip(bars, gaps):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{gap:.3f}", ha="center", va="bottom", fontsize=6)
plt.tight_layout()
save_path = ANALYSIS_DIR / "07_noise_robustness_gap.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()

# Ranking
ranked = sorted(zip(display_labels, gaps), key=lambda x: x[1])
print("\\nNoise robustness ranking (smaller = more robust):")
for rank, (name, gap) in enumerate(ranked, 1):
    print(f"  {rank:2d}. {name:<20} gap={gap:.4f}")
print(f"Saved -> {save_path}")
"""
    ))

    # --- Fig 8: PR curves
    cells.append(code(
"""# --- Fig 8: Precision-Recall curves (EIoU vs ECIoU vs best AEIoU, clean split)
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# Identify best AEIoU by clean mAP50-95
aeiou_entries = pivot95[pivot95.index.str.startswith("aeiou")]
best_aeiou = aeiou_entries["clean"].idxmax() if not aeiou_entries.empty else "aeiou_r0p3"

losses_for_pr = ["eiou", "eciou", best_aeiou]
pr_colors = {"eiou": "#E63946", "eciou": "#9B2226", best_aeiou: "#00B4D8"}

fig, ax = plt.subplots(figsize=(8, 6))

for loss_name in losses_for_pr:
    # Use first seed for PR curves
    seed = SEEDS[0]
    run_name = f"kvasir_yolo26n_{loss_name}_clean_s{seed}_e{EPOCHS}"
    weights = EXPERIMENTS / run_name / "weights" / "best.pt"
    if not weights.exists():
        print(f"[SKIP] {run_name} weights not found")
        continue

    model = YOLO(str(weights))
    val_res = model.val(data=str(SPLIT_CONFIGS["clean"]), verbose=False)

    # Extract PR data from Ultralytics results
    if hasattr(val_res.box, "curves") and "Precision" in val_res.box.curves:
        prec = val_res.box.curves["Precision"]
        rec  = val_res.box.curves["Recall"]
    elif hasattr(val_res.box, "p") and hasattr(val_res.box, "r"):
        prec = val_res.box.p.flatten()
        rec  = val_res.box.r.flatten()
    else:
        # Fallback: plot a simplified PR from multiple confidence thresholds
        print(f"  PR curve data not directly available for {loss_name}, using AP summary")
        continue

    color = pr_colors.get(loss_name, "#888")
    label = LOSS_LABELS.get(loss_name, loss_name)
    ax.plot(rec, prec, color=color, lw=2, label=f"{label} (AP={float(val_res.box.map50):.3f})")

ax.set_xlabel("Recall", fontsize=11)
ax.set_ylabel("Precision", fontsize=11)
ax.set_title("Precision-Recall Curves (clean split)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout()
save_path = ANALYSIS_DIR / "08_pr_curves.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved -> {save_path}")
"""
    ))

    # --- Fig 9: AP@thresholds
    cells.append(code(
"""# --- Fig 9: AP vs IoU threshold (EIoU vs ECIoU vs best AEIoU, clean split)
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

aeiou_entries = pivot95[pivot95.index.str.startswith("aeiou")]
best_aeiou = aeiou_entries["clean"].idxmax() if not aeiou_entries.empty else "aeiou_r0p3"
THRESHOLD_RANGE = np.arange(0.50, 1.00, 0.05)

results_by_thresh = {}
for loss_name in ["eiou", "eciou", best_aeiou]:
    seed = SEEDS[0]
    run_name = f"kvasir_yolo26n_{loss_name}_clean_s{seed}_e{EPOCHS}"
    weights = EXPERIMENTS / run_name / "weights" / "best.pt"
    if not weights.exists():
        print(f"[SKIP] {run_name}")
        continue
    model = YOLO(str(weights))
    aps = []
    for thresh in THRESHOLD_RANGE:
        val_res = model.val(data=str(SPLIT_CONFIGS["clean"]), iou=float(thresh), verbose=False)
        ap_val = float(val_res.box.map50) if hasattr(val_res.box, "map50") else float(val_res.box.maps[0])
        aps.append(ap_val)
    results_by_thresh[loss_name] = aps

colors = {"eiou": "#E63946", "eciou": "#9B2226", best_aeiou: "#00B4D8"}
fig, ax = plt.subplots(figsize=(8, 5))
for loss_name in results_by_thresh:
    label = LOSS_LABELS.get(loss_name, loss_name)
    ax.plot(THRESHOLD_RANGE, results_by_thresh[loss_name],
            marker="o", color=colors.get(loss_name, "#888"), lw=2, label=label)

ax.set_xlabel("IoU threshold", fontsize=11)
ax.set_ylabel("AP at threshold", fontsize=11)
ax.set_title("AP vs IoU Threshold (clean split)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
save_path = ANALYSIS_DIR / "09_ap_at_thresholds.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved -> {save_path}")
"""
    ))

    # --- Fig 10: Training stability
    cells.append(code(
"""# --- Fig 10: Training box loss stability (rolling std, window=3)
import matplotlib.pyplot as plt
import pandas as pd

LOSS_COL = "train/box_loss"
WINDOW = 3

aeiou_entries = pivot95[pivot95.index.str.startswith("aeiou")]
best_aeiou = aeiou_entries["clean"].idxmax() if not aeiou_entries.empty else "aeiou_r0p3"

# Compare EIoU, ECIoU, and best AEIoU on clean and high splits
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Training box loss stability (rolling std, window={WINDOW} epochs)",
             fontsize=13, fontweight="bold")

for ax, split in zip(axes, ["clean", "high"]):
    for loss_name in ["eiou", "eciou", best_aeiou]:
        color = PALETTE.get(loss_name, "#888")
        label = LOSS_LABELS.get(loss_name, loss_name)
        sub = df_all[(df_all["loss"] == loss_name) & (df_all["split"] == split)]
        if sub.empty or LOSS_COL not in sub.columns:
            continue
        avg = sub.groupby("epoch")[LOSS_COL].mean().reset_index().sort_values("epoch")
        loss_vals = avg[LOSS_COL].values
        epochs_arr = avg["epoch"].values
        rolling_std = pd.Series(loss_vals).rolling(WINDOW, min_periods=1).std().values
        ax.plot(epochs_arr, loss_vals, color=color, lw=2, label=label)
        ax.fill_between(epochs_arr,
                        loss_vals - rolling_std, loss_vals + rolling_std,
                        color=color, alpha=0.15)
    ax.set_title(f"Split: {split}", fontsize=10)
    ax.set_xlabel("Epoch"); ax.set_ylabel("box_loss")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

plt.tight_layout()
save_path = ANALYSIS_DIR / "10_training_stability.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved -> {save_path}")
"""
    ))

    # --- Fig 11: Box dimension calibration scatter
    cells.append(code(
"""# --- Fig 11: Box dimension calibration scatter (pred W,H vs GT W,H)
# For all 200 val images: compare predicted and GT box dimensions.
# Perfect calibration = points on y=x diagonal.
import matplotlib.pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path

aeiou_entries = pivot95[pivot95.index.str.startswith("aeiou")]
best_aeiou = aeiou_entries["clean"].idxmax() if not aeiou_entries.empty else "aeiou_r0p3"

val_img_dir = DATASET_ROOT / "valid" / "images"
val_lbl_dir = DATASET_ROOT / "valid" / "labels"

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Box Dimension Calibration: Predicted vs GT (clean split)",
             fontsize=13, fontweight="bold")

for ax, loss_name in zip(axes, ["eiou", "eciou", best_aeiou]):
    seed = SEEDS[0]
    run_name = f"kvasir_yolo26n_{loss_name}_clean_s{seed}_e{EPOCHS}"
    weights = EXPERIMENTS / run_name / "weights" / "best.pt"
    if not weights.exists():
        ax.set_title(f"{LOSS_LABELS.get(loss_name, loss_name)} — weights missing")
        continue
    model = YOLO(str(weights))

    gt_wh, pred_wh = [], []
    for lbl_path in sorted(val_lbl_dir.glob("*.txt")):
        img_path = val_img_dir / f"{lbl_path.stem}.jpg"
        if not img_path.exists(): continue
        # GT box
        line = lbl_path.read_text().strip().split("\\n")[0]
        _, cx, cy, bw, bh = [float(v) for v in line.split()]
        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]
        gt_w, gt_h = bw * W, bh * H
        gt_wh.append((gt_w, gt_h))
        # Pred box
        res = model(str(img_path), verbose=False)[0]
        if res.boxes and len(res.boxes.xyxy):
            pb = res.boxes.xyxy.cpu().numpy()[0]
            pw, ph = pb[2] - pb[0], pb[3] - pb[1]
            pred_wh.append((pw, ph))
        else:
            pred_wh.append((0, 0))

    gt_wh = np.array(gt_wh)
    pred_wh = np.array(pred_wh)
    # Scatter W and H together
    ax.scatter(gt_wh[:, 0], pred_wh[:, 0], alpha=0.5, s=15, c="#2A9D8F", label="Width")
    ax.scatter(gt_wh[:, 1], pred_wh[:, 1], alpha=0.5, s=15, c="#E63946", label="Height")
    max_val = max(gt_wh.max(), pred_wh.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", lw=1, label="y=x (perfect)")
    ax.set_xlabel("GT dimension (px)"); ax.set_ylabel("Predicted dimension (px)")
    ax.set_title(LOSS_LABELS.get(loss_name, loss_name), fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_aspect("equal")

plt.tight_layout()
save_path = ANALYSIS_DIR / "11_box_calibration.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved -> {save_path}")
"""
    ))

    cells.append(md(
"""**Figure 11 · Box Dimension Calibration**

*x-axis:* GT box dimension (W or H in pixels) · *y-axis:* Predicted dimension
*Dashed line:* y=x (perfect calibration)

**Expected pattern:** AEIoU points should cluster tighter to the diagonal,
especially for small polyps (bottom-left). EIoU may show systematic over-estimation
of box size on small objects because the enclosing-box normaliser under-penalises
size errors when the enclosing box is large.
"""
    ))

    # --- Fig 12: Statistical significance
    cells.append(code(
"""# --- Fig 12: Statistical significance tests
# Paired Wilcoxon signed-rank test on per-image prediction IoU
# comparing EIoU vs best AEIoU across all validation images.
import numpy as np
from ultralytics import YOLO
from scipy import stats
import cv2

aeiou_entries = pivot95[pivot95.index.str.startswith("aeiou")]
best_aeiou = aeiou_entries["clean"].idxmax() if not aeiou_entries.empty else "aeiou_r0p3"

val_img_dir = DATASET_ROOT / "valid" / "images"
val_lbl_dir = DATASET_ROOT / "valid" / "labels"

def _compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (aA + aB - inter + 1e-7)

# Collect per-image IoUs for each loss
per_image_ious = {}
for loss_name in ["eiou", "eciou", best_aeiou]:
    seed = SEEDS[0]
    run_name = f"kvasir_yolo26n_{loss_name}_clean_s{seed}_e{EPOCHS}"
    weights = EXPERIMENTS / run_name / "weights" / "best.pt"
    if not weights.exists():
        print(f"[SKIP] {run_name}")
        continue
    model = YOLO(str(weights))
    ious = []
    for lbl_path in sorted(val_lbl_dir.glob("*.txt")):
        img_path = val_img_dir / f"{lbl_path.stem}.jpg"
        if not img_path.exists(): continue
        line = lbl_path.read_text().strip().split("\\n")[0]
        _, cx, cy, bw, bh = [float(v) for v in line.split()]
        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]
        gt = np.array([(cx-bw/2)*W, (cy-bh/2)*H, (cx+bw/2)*W, (cy+bh/2)*H])
        res = model(str(img_path), verbose=False)[0]
        if res.boxes and len(res.boxes.xyxy):
            pb = res.boxes.xyxy.cpu().numpy()[0]
            ious.append(_compute_iou(pb, gt))
        else:
            ious.append(0.0)
    per_image_ious[loss_name] = np.array(ious)
    print(f"{LOSS_LABELS.get(loss_name, loss_name)}: mean IoU = {np.mean(ious):.4f}, n = {len(ious)}")

# Wilcoxon signed-rank test: EIoU vs best AEIoU
print("\\n=== Statistical Significance Tests (Wilcoxon signed-rank) ===")
if "eiou" in per_image_ious and best_aeiou in per_image_ious:
    stat, p = stats.wilcoxon(per_image_ious["eiou"], per_image_ious[best_aeiou])
    delta = np.mean(per_image_ious[best_aeiou]) - np.mean(per_image_ious["eiou"])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"EIoU vs {best_aeiou}: stat={stat:.1f}, p={p:.6f} ({sig})")
    print(f"  Mean IoU delta: {delta:+.4f} (positive = AEIoU better)")

if "eiou" in per_image_ious and "eciou" in per_image_ious:
    stat, p = stats.wilcoxon(per_image_ious["eiou"], per_image_ious["eciou"])
    delta = np.mean(per_image_ious["eciou"]) - np.mean(per_image_ious["eiou"])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"EIoU vs ECIoU: stat={stat:.1f}, p={p:.6f} ({sig})")
    print(f"  Mean IoU delta: {delta:+.4f} (positive = ECIoU better)")
"""
    ))

    # --- Fig 13: Cross-dataset lambda consistency
    cells.append(code(
"""# --- Fig 13: Cross-dataset lambda consistency (Kvasir vs DUO)
# Loads optimal lambda from notebook 02's results if available, otherwise
# provides a stub for manual comparison.
import json

print("=== Cross-Dataset Lambda Consistency ===\\n")

# This notebook's result
aeiou_clean = df_agg[(df_agg["split"] == "clean") & (df_agg["loss"].str.startswith("aeiou"))]
if not aeiou_clean.empty:
    kvasir_best = aeiou_clean.loc[aeiou_clean["map95_mean"].idxmax()]
    r_kvasir = kvasir_best["loss"].split("_r")[1].replace("p", ".")
    print(f"Kvasir-SEG optimal lambda: {r_kvasir}  (mAP95={kvasir_best['map95_mean']:.4f})")
else:
    r_kvasir = "?"
    print("Kvasir-SEG: no AEIoU results found")

# Try to load DUO results from notebook 02
duo_csv = PROJECT_DIR / "experiments" / "all_results_combined.csv"
if duo_csv.exists():
    import pandas as pd
    duo_df = pd.read_csv(duo_csv)
    if "loss" in duo_df.columns and "split" in duo_df.columns:
        duo_df.columns = duo_df.columns.str.strip()
        duo_final = duo_df.groupby("run_name").last().reset_index()
        duo_aeiou = duo_final[(duo_final["loss"].str.startswith("aeiou")) & (duo_final["split"] == "clean")]
        if "metrics/mAP50-95(B)" in duo_aeiou.columns and not duo_aeiou.empty:
            duo_best_idx = duo_aeiou["metrics/mAP50-95(B)"].idxmax()
            duo_best = duo_aeiou.loc[duo_best_idx]
            r_duo = duo_best["loss"].split("_r")[1].replace("p", ".")
            print(f"DUO optimal lambda:       {r_duo}  (mAP95={duo_best['metrics/mAP50-95(B)']:.4f})")

            if r_kvasir == r_duo:
                print(f"\\n*** MATCH: optimal lambda = {r_kvasir} on BOTH datasets ***")
                print("This supports a domain-agnostic default for amorphous objects.")
            else:
                print(f"\\nDifferent optima: Kvasir={r_kvasir}, DUO={r_duo}")
                print("The optimal lambda may be dataset-dependent.")
        else:
            print("DUO: no AEIoU results in combined CSV")
    else:
        print("DUO: CSV format unexpected")
else:
    print(f"DUO results not found at {duo_csv}")
    print("Run notebook 02 first to enable cross-dataset comparison.")
"""
    ))

    # --- Fig 14: Runtime overhead table
    cells.append(code(
"""# --- Fig 14: Runtime overhead per loss function
import json
import pandas as pd

print("=== Runtime Overhead ===\\n")

if MANIFEST_PATH.exists():
    manifest = json.loads(MANIFEST_PATH.read_text())
    timing = []
    for run_name, meta in manifest.items():
        if meta.get("elapsed_sec") and meta.get("status") == "complete":
            timing.append({
                "loss": meta.get("loss", "?"),
                "split": meta.get("split", "?"),
                "elapsed_sec": meta["elapsed_sec"],
                "sec_per_epoch": meta["elapsed_sec"] / meta.get("epochs", EPOCHS),
            })

    if timing:
        timing_df = pd.DataFrame(timing)
        summary = timing_df.groupby("loss").agg(
            mean_sec=("elapsed_sec", "mean"),
            mean_sec_per_epoch=("sec_per_epoch", "mean"),
            n_runs=("elapsed_sec", "count"),
        ).sort_values("mean_sec")

        print(summary.round(1).to_string())
        print(f"\\nFastest: {summary['mean_sec'].idxmin()} ({summary['mean_sec'].min():.1f}s)")
        print(f"Slowest: {summary['mean_sec'].idxmax()} ({summary['mean_sec'].max():.1f}s)")

        overhead = (summary["mean_sec"].max() - summary["mean_sec"].min()) / summary["mean_sec"].min() * 100
        print(f"Max overhead: {overhead:.1f}%")
        summary.to_csv(ANALYSIS_DIR / "runtime_overhead.csv")
        print(f"Saved -> {ANALYSIS_DIR / 'runtime_overhead.csv'}")
    else:
        print("No timing data in manifest.")
else:
    print("Manifest not found.")
"""
    ))

    # --- Fig 15: Per-prediction IoU distribution
    cells.append(code(
"""# --- Fig 15: Per-prediction IoU distribution (KDE, all val images)
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# Reuse per_image_ious from statistical test cell
fig, ax = plt.subplots(figsize=(8, 5))
x_range = np.linspace(0, 1, 200)

for loss_name in per_image_ious:
    data = per_image_ious[loss_name]
    if len(data) < 2:
        continue
    color = PALETTE.get(loss_name, "#888")
    label = LOSS_LABELS.get(loss_name, loss_name)
    kde = gaussian_kde(data, bw_method=0.15)
    ax.plot(x_range, kde(x_range), color=color, lw=2.5, label=label)
    ax.fill_between(x_range, kde(x_range), color=color, alpha=0.1)
    ax.axvline(np.mean(data), color=color, linestyle="--", lw=1,
               label=f"  mean={np.mean(data):.3f}")

ax.set_xlabel("IoU with ground truth", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Per-Prediction IoU Distribution (200 val images, clean split)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
save_path = ANALYSIS_DIR / "15_iou_distribution.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved -> {save_path}")
"""
    ))

    # --- Fig 16: Polyp-size stratified AP
    cells.append(code(
"""# --- Fig 16: Polyp-size stratified prediction quality (small/medium/large)
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

val_lbl_dir = DATASET_ROOT / "valid" / "labels"

# Stratify per-image IoUs by GT box area
areas = []
for lbl_path in sorted(val_lbl_dir.glob("*.txt")):
    line = lbl_path.read_text().strip().split("\\n")[0]
    _, cx, cy, bw, bh = [float(v) for v in line.split()]
    areas.append((lbl_path.stem, bw * bh))
areas.sort(key=lambda x: x[1])
n = len(areas)
strata = {
    "small":  set(s for s, _ in areas[:n//3]),
    "medium": set(s for s, _ in areas[n//3:2*n//3]),
    "large":  set(s for s, _ in areas[2*n//3:]),
}

# Build stratified results using per_image_ious (indexed by lbl_path order)
lbl_stems = [lbl_path.stem for lbl_path in sorted(val_lbl_dir.glob("*.txt"))]
stratified = {}
for loss_name in per_image_ious:
    stratified[loss_name] = {}
    ious = per_image_ious[loss_name]
    for stratum, stems in strata.items():
        indices = [i for i, s in enumerate(lbl_stems) if s in stems and i < len(ious)]
        stratified[loss_name][stratum] = np.mean([ious[i] for i in indices]) if indices else 0

stratum_names = ["small", "medium", "large"]
x = np.arange(len(stratum_names))
n_losses = len(per_image_ious)
w = 0.8 / max(n_losses, 1)
fig, ax = plt.subplots(figsize=(10, 5))

for i, loss_name in enumerate(per_image_ious):
    color = PALETTE.get(loss_name, "#888")
    label = LOSS_LABELS.get(loss_name, loss_name)
    vals = [stratified[loss_name][s] for s in stratum_names]
    ax.bar(x + i*w - (n_losses-1)*w/2, vals, w, color=color, alpha=0.85, label=label)

ax.set_xticks(x)
ax.set_xticklabels(["Small polyps\\n(bottom 33%)", "Medium polyps\\n(middle 33%)", "Large polyps\\n(top 33%)"])
ax.set_ylabel("Mean prediction IoU", fontsize=11)
ax.set_title("Polyp-Size Stratified Prediction Quality", fontsize=13, fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save_path = ANALYSIS_DIR / "16_size_stratified.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved -> {save_path}")
"""
    ))

    cells.append(md(
"""**Figure 16 · Size-Stratified Prediction Quality**

**Expected pattern:** AEIoU should gain most on **small polyps** — these have
highly irregular boundaries relative to their size, making EIoU's enclosing-box
normaliser overly lenient. AEIoU's target-dim normaliser scales the penalty
to the polyp itself, so small polyps get proportionally correct gradients.
"""
    ))

    # --- Fig 17: Failure case analysis
    cells.append(code(
"""# --- Fig 17: Failure case analysis — images where EIoU beats best AEIoU
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2, numpy as np

aeiou_entries = pivot95[pivot95.index.str.startswith("aeiou")]
best_aeiou = aeiou_entries["clean"].idxmax() if not aeiou_entries.empty else "aeiou_r0p3"

if "eiou" not in per_image_ious or best_aeiou not in per_image_ious:
    print("Skipping failure cases — per-image IoU data missing.")
else:
    eiou_ious  = per_image_ious["eiou"]
    aeiou_ious = per_image_ious[best_aeiou]

    # Find images where EIoU has higher IoU than AEIoU
    val_lbl_dir = DATASET_ROOT / "valid" / "labels"
    lbl_stems = [p.stem for p in sorted(val_lbl_dir.glob("*.txt"))]

    delta = eiou_ious - aeiou_ious
    failure_indices = np.argsort(delta)[::-1][:5]  # top 5 where EIoU > AEIoU

    if np.all(delta[failure_indices] <= 0):
        print("No failure cases found — AEIoU >= EIoU on all images!")
    else:
        fig, axes = plt.subplots(1, min(5, len(failure_indices)), figsize=(20, 4))
        if not hasattr(axes, "__len__"): axes = [axes]

        fig.suptitle(f"Failure Cases: Images Where EIoU Beats {best_aeiou}",
                     fontsize=13, fontweight="bold")

        for ax, idx in zip(axes, failure_indices):
            if idx >= len(lbl_stems):
                continue
            stem = lbl_stems[idx]
            img_path = DATASET_ROOT / "valid" / "images" / f"{stem}.jpg"
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_title(f"EIoU: {eiou_ious[idx]:.3f}  |  AEIoU: {aeiou_ious[idx]:.3f}\\n"
                         f"delta = {delta[idx]:+.3f}", fontsize=8)
            ax.axis("off")

        plt.tight_layout()
        save_path = ANALYSIS_DIR / "17_failure_cases.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved -> {save_path}")
        print(f"\\nFailure analysis: EIoU wins on {np.sum(delta > 0)}/{len(delta)} images "
              f"({np.sum(delta > 0)/len(delta)*100:.1f}%)")
"""
    ))

    cells.append(md(
"""**Figure 17 · Failure Cases**

Shows the images where EIoU produces higher per-prediction IoU than AEIoU.
Understanding failure modes is critical for an honest paper:
- Large, well-defined polyps with clear boundaries may not benefit from
  reduced shape penalty — EIoU's full penalty is appropriate
- Round polyps (AR near 1.0) have reliable extent labels, so lambda<1 is unnecessary
- If failure cases are all large/round, that confirms the hypothesis: AEIoU
  helps on amorphous objects, not on regular ones
"""
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # Bbox Visualisation
    # ─────────────────────────────────────────────────────────────────────────
    cells.append(md(
"""## Section 12 · Bounding Box Visualisation

Side-by-side comparison on 5 stratified demo images:
EIoU vs ECIoU vs best AEIoU, with GT box and per-prediction IoU labels.
"""
    ))

    cells.append(code(
"""# --- Select 5 stratified demo images + run inference from key models
import cv2, numpy as np, pickle
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

val_img_dir = DATASET_ROOT / "valid" / "images"
val_lbl_dir = DATASET_ROOT / "valid" / "labels"

# Stratified selection by GT box area
records = []
for img_path in sorted(val_img_dir.glob("*.jpg")):
    lbl_path = val_lbl_dir / f"{img_path.stem}.txt"
    if not lbl_path.exists(): continue
    line = lbl_path.read_text().strip().split("\\n")[0]
    _, cx, cy, bw, bh = [float(v) for v in line.split()]
    img = cv2.imread(str(img_path))
    H, W = img.shape[:2]
    x1=(cx-bw/2)*W; y1=(cy-bh/2)*H; x2=(cx+bw/2)*W; y2=(cy+bh/2)*H
    area = (x2-x1)*(y2-y1)
    records.append((img_path, (x1,y1,x2,y2), area))
records.sort(key=lambda r: r[2])
indices = np.linspace(0, len(records)-1, 5, dtype=int)
DEMO = [records[i] for i in indices]

aeiou_entries = pivot95[pivot95.index.str.startswith("aeiou")]
best_aeiou = aeiou_entries["clean"].idxmax() if not aeiou_entries.empty else "aeiou_r0p3"
vis_losses = ["eiou", "eciou", best_aeiou]

# Run inference
inference = {}
for loss_name in vis_losses:
    seed = SEEDS[0]
    run_name = f"kvasir_yolo26n_{loss_name}_clean_s{seed}_e{EPOCHS}"
    weights = EXPERIMENTS / run_name / "weights" / "best.pt"
    if not weights.exists():
        print(f"[SKIP] {run_name}")
        continue
    model = YOLO(str(weights))
    preds = {}
    for img_path, gt, area in DEMO:
        res = model(str(img_path), verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes else np.zeros((0,4))
        preds[img_path.name] = boxes
    inference[loss_name] = preds

# Draw comparison grid
n_rows, n_cols = len(DEMO), len(vis_losses)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
fig.suptitle("Bbox Comparison: EIoU vs ECIoU vs Best AEIoU",
             fontsize=13, fontweight="bold", y=1.01)

for row, (img_path, gt_box, area) in enumerate(DEMO):
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    gt_np = np.array(gt_box)

    for col, loss_name in enumerate(vis_losses):
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.imshow(img)
        # GT box — white dashed
        x1,y1,x2,y2 = gt_np
        ax.add_patch(mpatches.Rectangle((x1,y1), x2-x1, y2-y1,
                     linewidth=1.5, edgecolor="white", facecolor="none", linestyle="--"))
        # Predictions
        color = PALETTE.get(loss_name, "#888")
        boxes = inference.get(loss_name, {}).get(img_path.name, np.zeros((0,4)))
        for pb in boxes:
            iou = _compute_iou(pb, gt_np)
            rx1,ry1,rx2,ry2 = pb
            ax.add_patch(mpatches.Rectangle((rx1,ry1), rx2-rx1, ry2-ry1,
                         linewidth=2, edgecolor=color, facecolor="none"))
            ax.text(rx1, ry1-5, f"IoU={iou:.2f}", color=color, fontsize=8,
                    fontweight="bold", bbox=dict(facecolor="black", alpha=0.6, pad=1))
        if row == 0:
            ax.set_title(LOSS_LABELS.get(loss_name, loss_name), fontsize=10, fontweight="bold", color=color)
        if col == 0:
            ax.set_ylabel(f"area={area:.0f}px2", fontsize=8)
        ax.axis("off")

plt.tight_layout()
save_path = ANALYSIS_DIR / "18_bbox_comparison.png"
plt.savefig(save_path, dpi=120, bbox_inches="tight")
plt.show()
print(f"Saved -> {save_path}")
"""
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # Robustness Ranking
    # ─────────────────────────────────────────────────────────────────────────
    cells.append(code(
"""# --- Noise robustness ranking table (all losses)
import pandas as pd

all_losses = BASELINE_LOSS_NAMES + [f"aeiou_r{_fmt_r(r)}" for r in AEIOU_RIGIDITIES]
ranking = []
for loss_name in all_losses:
    if loss_name not in pivot95.index:
        continue
    clean_map = pivot95.loc[loss_name, "clean"]
    low_map   = pivot95.loc[loss_name, "low"]
    high_map  = pivot95.loc[loss_name, "high"]
    ratio     = high_map / clean_map if clean_map > 0 else 0
    ranking.append({
        "loss": loss_name, "label": LOSS_LABELS.get(loss_name, loss_name),
        "clean": clean_map, "low": low_map, "high": high_map,
        "robust_ratio": ratio,
    })

rank_df = pd.DataFrame(ranking).sort_values("robust_ratio", ascending=False).reset_index(drop=True)
rank_df.index += 1

print("Noise Robustness Ranking (mAP_high / mAP_clean — higher = more robust):")
print(rank_df.to_string())
rank_df.to_csv(ANALYSIS_DIR / "robustness_ranking.csv")
champion = rank_df.iloc[0]
print(f"\\nMost robust: {champion['label']}  ratio={champion['robust_ratio']:.4f}")
"""
    ))

    # ─────────────────────────────────────────────────────────────────────────
    # Summary & Conclusions
    # ─────────────────────────────────────────────────────────────────────────
    cells.append(md(
"""## Section 13 · Summary & Conclusions

### Key Findings

| Analysis | What It Tests | Supports AEIoU? |
|---|---|---|
| mAP50-95 (Fig 2) | Localisation quality vs ALL baselines | If AEIoU > all 6 baselines |
| mAP50 (Fig 2b) | Detection quality | If AEIoU improves detection, not just boxes |
| Lambda curve (Fig 3) | Optimal lambda and whether it beats all baselines | Peak above all reference lines |
| Lambda heatmap (Fig 4) | Stability of optimal lambda across noise | Same peak across splits |
| Learning curves (Fig 5) | Convergence behaviour | AEIoU learns faster or to higher mAP |
| Convergence (Fig 6) | Speed to 90% mAP | Lower bar = faster |
| Robustness (Fig 7) | mAP degradation under noise | Smaller gap = more robust |
| PR curves (Fig 8) | Detection-recall trade-off | Higher curve = better |
| AP@thresholds (Fig 9) | Consistency across IoU thresholds | Above baselines at all thresholds |
| Stability (Fig 10) | Training smoothness | Narrower band = smoother |
| Box calibration (Fig 11) | Dimensional accuracy | Tighter to y=x diagonal |
| Significance (Fig 12) | Statistical validity | p < 0.05 |
| Cross-dataset (Fig 13) | Domain-agnostic lambda | Same optimal lambda as DUO |
| Runtime (Fig 14) | Computational cost | < 5% overhead |
| IoU distribution (Fig 15) | Overall box quality | Right-shifted KDE |
| Size-stratified (Fig 16) | Where gains concentrate | Largest on small polyps |
| Failure cases (Fig 17) | Honesty — where does AEIoU fail? | Large/round polyps |

### Recommended settings for Kvasir-SEG polyp detection
```
loss = AEIoULoss(rigidity=<optimal_lambda>, reduction="none")
```

### For publication
- Set `SEEDS = [42, 123, 456]` and re-run to get mean +/- std
- All analyses automatically handle multi-seed aggregation
- The statistical significance test uses paired Wilcoxon on per-image IoU
"""
    ))

    cells.append(code(
"""# --- Save all artifacts and print manifest summary
import json, os

print("=== Experiment Artifact Summary ===\\n")
print(f"Experiments dir: {EXPERIMENTS}")
print(f"Analysis dir:    {ANALYSIS_DIR}")

print("\\nSaved figures:")
for f in sorted(ANALYSIS_DIR.glob("*.png")):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:<45} {size_kb:>6.1f} KB")

print("\\nSaved tables:")
for f in sorted(ANALYSIS_DIR.glob("*.csv")):
    print(f"  {f.name}")

if MANIFEST_PATH.exists():
    manifest = json.loads(MANIFEST_PATH.read_text())
    complete = sum(1 for v in manifest.values() if v.get("status") == "complete")
    failed   = sum(1 for v in manifest.values() if v.get("status") == "failed")
    print(f"\\nManifest: {len(manifest)} runs  |  {complete} complete  |  {failed} failed")

print("\\nAll artifacts saved.")
"""
    ))

    cells.append(code(
"""# --- Final sync: push analysis figures and manifest to Drive
import shutil

if DRIVE_AVAILABLE:
    drive_analysis = DRIVE_EXPERIMENTS / "analysis"
    drive_analysis.mkdir(parents=True, exist_ok=True)
    if ANALYSIS_DIR.exists():
        shutil.copytree(str(ANALYSIS_DIR), str(drive_analysis), dirs_exist_ok=True)
        n_figs = len(list(drive_analysis.glob("*.png")))
        print(f"Analysis figures synced to Drive: {n_figs} PNGs")

    for fname in ["manifest.json", "all_results_combined.csv"]:
        src = EXPERIMENTS / fname
        if src.exists():
            shutil.copy2(str(src), str(DRIVE_EXPERIMENTS / fname))
            print(f"  Synced {fname}")

    print(f"\\nAll artifacts backed up to: {DRIVE_EXPERIMENTS}")
    n_total = sum(1 for _ in DRIVE_EXPERIMENTS.rglob("*") if _.is_file())
    print(f"Total files on Drive: {n_total}")
else:
    print("Drive not mounted — skipping final sync.")
    print("Tip: Run mount_drive() and re-run this cell to back up manually.")

print("\\nNotebook 03 complete.")
"""
    ))

    return cells
