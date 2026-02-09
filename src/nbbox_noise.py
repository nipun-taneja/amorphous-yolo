import os
import math
import random
from pathlib import Path
import shutil


def jitter_box(xc, yc, w, h,
               sigma_pos_x, sigma_pos_y,
               sigma_w, sigma_h):
    """
    Apply synthetic annotation noise to a single normalized YOLO box.
    All coords are in [0,1] relative to image width/height.
    """
    # positional noise in units of width/height
    dx = random.gauss(0.0, sigma_pos_x) * w
    dy = random.gauss(0.0, sigma_pos_y) * h

    # multiplicative size noise (log-normal style)
    fx = math.exp(random.gauss(0.0, sigma_w))
    fy = math.exp(random.gauss(0.0, sigma_h))

    xc_noisy = xc + dx
    yc_noisy = yc + dy
    w_noisy = w * fx
    h_noisy = h * fy

    # clip to normalized [0,1] and keep box valid
    eps = 1e-6
    w_noisy = max(eps, min(1.0, w_noisy))
    h_noisy = max(eps, min(1.0, h_noisy))

    xc_noisy = max(w_noisy / 2, min(1.0 - w_noisy / 2, xc_noisy))
    yc_noisy = max(h_noisy / 2, min(1.0 - h_noisy / 2, yc_noisy))

    return xc_noisy, yc_noisy, w_noisy, h_noisy


def make_noisy_labels(src_labels_dir: Path,
                      dst_labels_dir: Path,
                      sigma_pos_x: float,
                      sigma_pos_y: float,
                      sigma_w: float,
                      sigma_h: float):
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    for txt_path in sorted(src_labels_dir.glob("*.txt")):
        lines_out = []
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, xc, yc, w, h = parts
                xc = float(xc)
                yc = float(yc)
                w = float(w)
                h = float(h)

                xc_n, yc_n, w_n, h_n = jitter_box(
                    xc, yc, w, h,
                    sigma_pos_x, sigma_pos_y,
                    sigma_w, sigma_h,
                )

                lines_out.append(
                    f"{cls_id} {xc_n:.6f} {yc_n:.6f} {w_n:.6f} {h_n:.6f}\n"
                )

        out_path = dst_labels_dir / txt_path.name
        with open(out_path, "w") as f:
            f.writelines(lines_out)


def make_symlink_or_copy_images(src_images_dir: Path, dst_images_dir: Path):
    """
    Create dst_images_dir containing either symlinks or copies of images.
    Symlinks keep disk usage low; fallback to copy if symlinks not allowed.
    """
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    for img_path in sorted(src_images_dir.glob("*")):
        if not img_path.is_file():
            continue
        dst_path = dst_images_dir / img_path.name
        if dst_path.exists():
            continue
        try:
            # symlink for efficiency
            dst_path.symlink_to(img_path)
        except OSError:
            # fallback: copy
            shutil.copy2(img_path, dst_path)


def build_duo_noise_splits(root: Path):
    """
    root: /content/amorphous-yolo/datasets/DUO_dataset
    Creates:
      - valid      (existing, clean)
      - valid_low  (new)
      - valid_high (new)
    Each with images/ and labels/.
    """
    src_images = root / "valid" / "images"
    src_labels = root / "valid" / "labels"

    # Targets
    valid_low_images = root / "valid_low" / "images"
    valid_low_labels = root / "valid_low" / "labels"

    valid_high_images = root / "valid_high" / "images"
    valid_high_labels = root / "valid_high" / "labels"

    # Share images by symlink/copy
    make_symlink_or_copy_images(src_images, valid_low_images)
    make_symlink_or_copy_images(src_images, valid_high_images)

    # Noise configs:
    # Low noise
    make_noisy_labels(
        src_labels_dir=src_labels,
        dst_labels_dir=valid_low_labels,
        sigma_pos_x=0.02,
        sigma_pos_y=0.02,
        sigma_w=0.02,
        sigma_h=0.02,
    )

    # High noise
    make_noisy_labels(
        src_labels_dir=src_labels,
        dst_labels_dir=valid_high_labels,
        sigma_pos_x=0.05,
        sigma_pos_y=0.05,
        sigma_w=0.05,
        sigma_h=0.10,
    )


if __name__ == "__main__":
    ROOT = Path("/content/amorphous-yolo/datasets/DUO_dataset")
    build_duo_noise_splits(ROOT)
