from pathlib import Path
import math
import random

def _quad_to_xywh(coords):
    """
    coords: list/tuple of 8 floats [x1,y1,x2,y2,x3,y3,x4,y4] in normalized [0,1].
    Return xc,yc,w,h (normalized).
    """
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = max(1e-6, x_max - x_min)
    h = max(1e-6, y_max - y_min)
    xc = x_min + w / 2.0
    yc = y_min + h / 2.0
    return xc, yc, w, h


def _xywh_to_quad(xc, yc, w, h):
    """
    Given xc,yc,w,h in [0,1], return axis-aligned quad
    [x1,y1,x2,y2,x3,y3,x4,y4] (clockwise).
    """
    x_min = xc - w / 2.0
    x_max = xc + w / 2.0
    y_min = yc - h / 2.0
    y_max = yc + h / 2.0
    return [
        x_min, y_min,
        x_max, y_min,
        x_max, y_max,
        x_min, y_max,
    ]


def make_noisy_labels(src_labels_dir: Path,
                      dst_labels_dir: Path,
                      sigma_pos_x: float,
                      sigma_pos_y: float,
                      sigma_w: float,
                      sigma_h: float,
                      rng_seed: int | None = None):
    """
    DUO-style polygon labels:
      cls x1 y1 x2 y2 x3 y3 x4 y4

    We:
      - convert quad -> (xc,yc,w,h)
      - apply noise in (xc,yc,w,h)
      - convert back to quad and write same format.
    """
    if rng_seed is not None:
        random.seed(rng_seed)

    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    for txt_path in sorted(src_labels_dir.glob("*.txt")):
        lines_out = []
        with open(txt_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    # skip malformed lines
                    continue

                cls_id = parts[0]
                coords = list(map(float, parts[1:9]))  # x1..y4

                xc, yc, w, h = _quad_to_xywh(coords)

                # jitter in (xc,yc,w,h)
                dx = random.gauss(0.0, sigma_pos_x) * w
                dy = random.gauss(0.0, sigma_pos_y) * h
                fx = math.exp(random.gauss(0.0, sigma_w))
                fy = math.exp(random.gauss(0.0, sigma_h))

                xc_n = xc + dx
                yc_n = yc + dy
                w_n = w * fx
                h_n = h * fy

                # clip to [0,1] and keep valid
                eps = 1e-6
                w_n = max(eps, min(1.0, w_n))
                h_n = max(eps, min(1.0, h_n))
                xc_n = max(w_n / 2, min(1.0 - w_n / 2, xc_n))
                yc_n = max(h_n / 2, min(1.0 - h_n / 2, yc_n))

                quad_n = _xywh_to_quad(xc_n, yc_n, w_n, h_n)
                quad_n_str = " ".join(f"{v:.6f}" for v in quad_n)
                lines_out.append(f"{cls_id} {quad_n_str}\n")

        out_path = dst_labels_dir / txt_path.name
        with open(out_path, "w") as f:
            f.writelines(lines_out)
