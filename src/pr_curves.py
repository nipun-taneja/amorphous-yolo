"""Per-loss PR-curve CSV exports from Ultralytics validation runs.

Ultralytics' built-in `model.val()` exposes a confidence sweep at IoU=0.5 only
(`metrics.box.p_curve` / `r_curve` / `f1_curve`). To get PR curves at the other
IoU thresholds (0.55, 0.6, ..., 0.95) we slice the raw TP matrix returned by
the validator and call `ap_per_class` once per IoU bin.

Output schema (long-format CSV per run):
    loss, iou_threshold, conf_threshold, precision, recall, f1
"""
from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _make_subset_yaml(yaml_path: Path, n_images: int, tmp_dir: Path) -> Path:
    """Copy the first n_images of the val set (and matching labels) into
    tmp_dir and write a temporary YAML pointing at the subset.
    """
    import yaml as _yaml

    cfg = _yaml.safe_load(Path(yaml_path).read_text())
    base = Path(cfg.get("path", ""))
    val_rel = cfg.get("val", "valid/images")
    val_dir = (base / val_rel) if str(base) else Path(val_rel)
    label_dir = val_dir.parent / "labels"

    sub_img = tmp_dir / "valid" / "images"
    sub_lbl = tmp_dir / "valid" / "labels"
    sub_img.mkdir(parents=True, exist_ok=True)
    sub_lbl.mkdir(parents=True, exist_ok=True)

    images = sorted(val_dir.glob("*.jpg")) + sorted(val_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"no .jpg/.png images under {val_dir}")
    images = images[:n_images]
    for ip in images:
        shutil.copy2(ip, sub_img / ip.name)
        lp = label_dir / f"{ip.stem}.txt"
        if lp.exists():
            shutil.copy2(lp, sub_lbl / lp.name)

    new_cfg = dict(cfg)
    new_cfg["path"] = str(tmp_dir)
    new_cfg["val"] = "valid/images"
    new_cfg["train"] = "valid/images"  # validator still expects a train key

    new_yaml = tmp_dir / "subset.yaml"
    new_yaml.write_text(_yaml.safe_dump(new_cfg))
    return new_yaml


@contextmanager
def _maybe_subset_yaml(yaml_path: Path, n_images: int | None):
    if n_images is None:
        yield Path(yaml_path)
        return
    tmp_dir = Path(tempfile.mkdtemp(prefix="pr_curve_subset_"))
    try:
        yield _make_subset_yaml(Path(yaml_path), n_images, tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _concat_stats(stats: dict) -> dict:
    """Ultralytics stores per-batch arrays as lists in `validator.stats`.
    Concatenate to flat ndarrays. Idempotent: already-flat arrays pass through.
    """
    out = {}
    for k, v in stats.items():
        if isinstance(v, list):
            out[k] = np.concatenate(v, 0) if len(v) else np.array([])
        else:
            out[k] = np.asarray(v)
    return out


def _run_validator(weights: Path, data_yaml: Path, conf: float,
                   iou_nms: float, imgsz: int, device):
    """Run Ultralytics' DetectionValidator and return it after `__call__`.

    Mirrors the pattern already used in notebook 07 cell `vfy-metrics-fn`.
    """
    from ultralytics.cfg import get_cfg, DEFAULT_CFG
    from ultralytics.models.yolo.detect import DetectionValidator

    args = get_cfg(DEFAULT_CFG, dict(
        model=str(weights), data=str(data_yaml),
        conf=conf, iou=iou_nms, imgsz=imgsz, device=device,
        verbose=False, save_json=False, plots=False, split="val",
    ))
    v = DetectionValidator(args=args)
    v()
    return v


def export_pr_curves_csv(
    run_dir: Path | str,
    data_yaml: Path | str,
    loss_name: str,
    out_csv: Path | str,
    *,
    conf: float = 0.001,
    iou_nms: float = 0.6,
    imgsz: int = 640,
    device: int | str = 0,
    max_val_images: int | None = None,
) -> Path:
    """Export PR-sweep CSV for one trained run.

    Args:
        run_dir: Ultralytics run directory containing weights/best.pt.
        data_yaml: Same dataset yaml passed to training.
        loss_name: Identifier emitted in the `loss` column (e.g. "aeiou_r0p5").
        out_csv: Destination CSV path.
        conf: Confidence pre-filter (keep low to retain the right tail).
        iou_nms: NMS IoU. Distinct from eval IoU thresholds, which are fixed
            at np.linspace(0.5, 0.95, 10) by Ultralytics.
        imgsz, device: Forwarded to the validator.
        max_val_images: Smoke-test knob. If set, copies only the first N val
            images (and their labels) into a temp dir and validates against
            that subset. Curves on tiny subsets are noisy but the pipeline
            runs end-to-end in seconds.

    Returns:
        Path to the written CSV.
    """
    from ultralytics.utils.metrics import ap_per_class

    run_dir = Path(run_dir)
    weights = run_dir / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    with _maybe_subset_yaml(Path(data_yaml), max_val_images) as resolved_yaml:
        v = _run_validator(weights, resolved_yaml, conf, iou_nms, imgsz, device)

    stats = _concat_stats(dict(v.stats))
    tp = np.asarray(stats["tp"])
    conf_arr = np.asarray(stats["conf"])
    pred_cls = np.asarray(stats["pred_cls"])
    target_cls = np.asarray(stats["target_cls"])
    iouv = v.iouv.detach().cpu().numpy() if hasattr(v.iouv, "detach") else np.asarray(v.iouv)

    if tp.ndim != 2 or tp.shape[1] != len(iouv):
        raise RuntimeError(
            f"unexpected tp shape {tp.shape}; expected (n_preds, {len(iouv)})"
        )

    rows = []
    for k, iou_thr in enumerate(iouv):
        tp_slice = tp[:, k:k + 1]
        result = ap_per_class(tp_slice, conf_arr, pred_cls, target_cls, plot=False)
        # ap_per_class returns: tp, fp, p, r, f1, ap, unique_classes,
        #                       p_curve, r_curve, f1_curve, x, prec_values
        p_curve, r_curve, f1_curve, px = result[7], result[8], result[9], result[10]

        # p_curve / r_curve / f1_curve: shape (nc, n_px). Average across classes
        # so single-class polyp datasets collapse to (n_px,).
        p_mean = p_curve.mean(axis=0) if p_curve.ndim == 2 else p_curve
        r_mean = r_curve.mean(axis=0) if r_curve.ndim == 2 else r_curve
        f_mean = f1_curve.mean(axis=0) if f1_curve.ndim == 2 else f1_curve

        for i in range(len(px)):
            rows.append((
                loss_name,
                round(float(iou_thr), 2),
                float(px[i]),
                float(p_mean[i]),
                float(r_mean[i]),
                float(f_mean[i]),
            ))

    df = pd.DataFrame(rows, columns=[
        "loss", "iou_threshold", "conf_threshold", "precision", "recall", "f1",
    ])
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def export_all(
    runs: dict[str, Path | str],
    data_yaml: Path | str,
    out_dir: Path | str,
    master_csv: Path | str | None = None,
    **kwargs,
) -> Path | None:
    """Convenience wrapper: export one CSV per run, optionally concatenate.

    Args:
        runs: mapping of loss_name -> run_dir.
        data_yaml: dataset yaml.
        out_dir: per-run CSVs land at out_dir/<loss_name>_pr_curve.csv.
        master_csv: if given, also writes a concatenated CSV of every run.
        **kwargs: forwarded to export_pr_curves_csv (conf, iou_nms, imgsz, device).

    Returns:
        Path to the master CSV if requested, else None.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for loss_name, run_dir in runs.items():
        csv_path = out_dir / f"{loss_name}_pr_curve.csv"
        export_pr_curves_csv(run_dir, data_yaml, loss_name, csv_path, **kwargs)
        frames.append(pd.read_csv(csv_path))

    if master_csv is not None and frames:
        master = pd.concat(frames, ignore_index=True)
        master_csv = Path(master_csv)
        master_csv.parent.mkdir(parents=True, exist_ok=True)
        master.to_csv(master_csv, index=False)
        return master_csv
    return None
