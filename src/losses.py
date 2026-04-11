import math
import torch
import torch.nn as nn


def _bbox_iou_xyxy(pred_boxes, target_boxes, eps: float = 1e-7):
    """
    Basic IoU in xyxy format.
    pred_boxes, target_boxes: (..., 4) with [x1, y1, x2, y2].
    """
    # Intersection box
    x1 = torch.max(pred_boxes[..., 0], target_boxes[..., 0])
    y1 = torch.max(pred_boxes[..., 1], target_boxes[..., 1])
    x2 = torch.min(pred_boxes[..., 2], target_boxes[..., 2])
    y2 = torch.min(pred_boxes[..., 3], target_boxes[..., 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    # Areas
    area_p = (pred_boxes[..., 2] - pred_boxes[..., 0]).clamp(min=0) * \
             (pred_boxes[..., 3] - pred_boxes[..., 1]).clamp(min=0)
    area_t = (target_boxes[..., 2] - target_boxes[..., 0]).clamp(min=0) * \
             (target_boxes[..., 3] - target_boxes[..., 1]).clamp(min=0)

    # Union
    union = area_p + area_t - inter + eps
    return inter / union


def _enclosing_box(pred_boxes, target_boxes, eps: float = 1e-7):
    """
    Returns the dimensions of the smallest enclosing box for each pair.
    Returns: (cx1, cy1, cx2, cy2, cw, ch, c2)
      cw, ch = enclosing width/height
      c2     = enclosing diagonal squared
    """
    cx1 = torch.min(pred_boxes[..., 0], target_boxes[..., 0])
    cy1 = torch.min(pred_boxes[..., 1], target_boxes[..., 1])
    cx2 = torch.max(pred_boxes[..., 2], target_boxes[..., 2])
    cy2 = torch.max(pred_boxes[..., 3], target_boxes[..., 3])
    cw = (cx2 - cx1).clamp(min=eps)
    ch = (cy2 - cy1).clamp(min=eps)
    c2 = cw * cw + ch * ch
    return cx1, cy1, cx2, cy2, cw, ch, c2


def _apply_reduction(loss, reduction):
    """Apply reduction to a per-element loss tensor."""
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss  # "none"


# ---------------------------------------------------------------------------
# Loss classes
# ---------------------------------------------------------------------------

class IoULoss(nn.Module):
    """
    Baseline IoU loss: L = 1 - IoU.

    Does not account for center distance, aspect ratio, or box size — purely
    measures overlap. Included as the simplest possible baseline.
    """
    def __init__(self, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred_boxes, target_boxes):
        iou = _bbox_iou_xyxy(pred_boxes, target_boxes, eps=self.eps)
        loss = 1.0 - iou
        return _apply_reduction(loss, self.reduction)


class GIoULoss(nn.Module):
    """
    Generalized IoU loss (Rezatofighi et al., 2019).

    L = 1 - IoU + (C - U) / C

    where C is the area of the smallest enclosing box and U is the union area.
    The penalty term (C - U)/C pushes the predicted box toward the target even
    when they do not overlap (IoU=0), fixing the zero-gradient problem of plain IoU.
    """
    def __init__(self, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred_boxes, target_boxes):
        eps = self.eps

        # IoU and union
        x1 = torch.max(pred_boxes[..., 0], target_boxes[..., 0])
        y1 = torch.max(pred_boxes[..., 1], target_boxes[..., 1])
        x2 = torch.min(pred_boxes[..., 2], target_boxes[..., 2])
        y2 = torch.min(pred_boxes[..., 3], target_boxes[..., 3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

        area_p = (pred_boxes[..., 2] - pred_boxes[..., 0]).clamp(min=0) * \
                 (pred_boxes[..., 3] - pred_boxes[..., 1]).clamp(min=0)
        area_t = (target_boxes[..., 2] - target_boxes[..., 0]).clamp(min=0) * \
                 (target_boxes[..., 3] - target_boxes[..., 1]).clamp(min=0)
        union = area_p + area_t - inter + eps
        iou = inter / union

        # Enclosing box area
        _, _, _, _, cw, ch, _ = _enclosing_box(pred_boxes, target_boxes, eps=eps)
        C = cw * ch + eps

        giou = iou - (C - union) / C
        loss = 1.0 - giou
        return _apply_reduction(loss, self.reduction)


class DIoULoss(nn.Module):
    """
    Distance IoU loss (Zheng et al., 2020).

    L = 1 - IoU + ρ²(b, b^gt) / c²

    where ρ is the Euclidean distance between box centers and c is the diagonal
    of the smallest enclosing box. Penalises center misalignment regardless of
    overlap, accelerating convergence.
    """
    def __init__(self, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred_boxes, target_boxes):
        eps = self.eps
        iou = _bbox_iou_xyxy(pred_boxes, target_boxes, eps=eps)

        # Box centers
        pcx = (pred_boxes[..., 0] + pred_boxes[..., 2]) * 0.5
        pcy = (pred_boxes[..., 1] + pred_boxes[..., 3]) * 0.5
        tcx = (target_boxes[..., 0] + target_boxes[..., 2]) * 0.5
        tcy = (target_boxes[..., 1] + target_boxes[..., 3]) * 0.5

        rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
        _, _, _, _, _, _, c2 = _enclosing_box(pred_boxes, target_boxes, eps=eps)

        loss = 1.0 - iou + rho2 / (c2 + eps)
        return _apply_reduction(loss, self.reduction)


class CIoULoss(nn.Module):
    """
    Complete IoU loss (Zheng et al., 2020).

    L = 1 - IoU + ρ²/c² + α·v

    where:
        v = (4/π²) · (arctan(wt/ht) − arctan(wp/hp))²   [aspect ratio consistency]
        α = v / (1 − IoU + v + ε)                         [trade-off weight]

    The aspect ratio term v enforces that the predicted box has the same w/h ratio
    as the ground truth. This is the Ultralytics default loss for YOLO models.

    ⚠ Limitation for amorphous objects: v penalises any deviation from the target
    aspect ratio. Amorphous objects (scallop, holothurian) have highly variable and
    annotation-dependent aspect ratios, making this penalty misleading.
    """
    def __init__(self, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred_boxes, target_boxes):
        eps = self.eps
        iou = _bbox_iou_xyxy(pred_boxes, target_boxes, eps=eps)

        # Centers
        pcx = (pred_boxes[..., 0] + pred_boxes[..., 2]) * 0.5
        pcy = (pred_boxes[..., 1] + pred_boxes[..., 3]) * 0.5
        tcx = (target_boxes[..., 0] + target_boxes[..., 2]) * 0.5
        tcy = (target_boxes[..., 1] + target_boxes[..., 3]) * 0.5

        rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
        _, _, _, _, _, _, c2 = _enclosing_box(pred_boxes, target_boxes, eps=eps)

        # Box dimensions
        pw = (pred_boxes[..., 2] - pred_boxes[..., 0]).clamp(min=eps)
        ph = (pred_boxes[..., 3] - pred_boxes[..., 1]).clamp(min=eps)
        tw = (target_boxes[..., 2] - target_boxes[..., 0]).clamp(min=eps)
        th = (target_boxes[..., 3] - target_boxes[..., 1]).clamp(min=eps)

        # Aspect ratio consistency term
        v = (4.0 / (math.pi ** 2)) * (torch.atan2(tw, th) - torch.atan2(pw, ph)) ** 2
        with torch.no_grad():
            alpha = v / (1.0 - iou + v + eps)

        loss = 1.0 - iou + rho2 / (c2 + eps) + alpha * v
        return _apply_reduction(loss, self.reduction)


class EIoULoss(nn.Module):
    """
    Efficient IoU loss — Zheng et al., 2021 ("Enhancing Geometric Factors").

    L = 1 - IoU + ρ²/c² + (pw − tw)²/cw² + (ph − th)²/ch²

    Replaces CIoU's indirect aspect-ratio term (v) with direct, independently
    weighted width and height error terms, each normalised by the corresponding
    enclosing-box dimension. This decouples width and height penalties, giving
    the optimiser independent gradients for each axis.

    Normalisation by enclosing-box dimensions (cw, ch) makes the penalty
    context-aware: a 10px width error matters more in a tight enclosing box
    than in a large one.
    """
    def __init__(self, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred_boxes, target_boxes):
        eps = self.eps
        iou = _bbox_iou_xyxy(pred_boxes, target_boxes, eps=eps)

        # Centers
        pcx = (pred_boxes[..., 0] + pred_boxes[..., 2]) * 0.5
        pcy = (pred_boxes[..., 1] + pred_boxes[..., 3]) * 0.5
        tcx = (target_boxes[..., 0] + target_boxes[..., 2]) * 0.5
        tcy = (target_boxes[..., 1] + target_boxes[..., 3]) * 0.5

        rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
        _, _, _, _, cw, ch, c2 = _enclosing_box(pred_boxes, target_boxes, eps=eps)

        # Box dimensions
        pw = (pred_boxes[..., 2] - pred_boxes[..., 0]).clamp(min=eps)
        ph = (pred_boxes[..., 3] - pred_boxes[..., 1]).clamp(min=eps)
        tw = (target_boxes[..., 2] - target_boxes[..., 0]).clamp(min=eps)
        th = (target_boxes[..., 3] - target_boxes[..., 1]).clamp(min=eps)

        # Size terms normalised by enclosing-box dimensions
        w_term = (pw - tw) ** 2 / (cw ** 2 + eps)
        h_term = (ph - th) ** 2 / (ch ** 2 + eps)

        loss = 1.0 - iou + rho2 / (c2 + eps) + w_term + h_term
        return _apply_reduction(loss, self.reduction)


class ECIoULoss(nn.Module):
    """
    Extended Complete IoU loss.

    L = 1 - IoU + ρ²/c² + (pw − tw)²/max(pw,tw)² + (ph − th)²/max(ph,th)²

    Similar to EIoU but normalises size errors by the maximum of (predicted,
    target) dimension rather than the enclosing-box dimension. This makes the
    penalty relative to the larger of the two boxes, which can be more stable
    when the enclosing box is much larger than either box.
    """
    def __init__(self, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred_boxes, target_boxes):
        eps = self.eps
        iou = _bbox_iou_xyxy(pred_boxes, target_boxes, eps=eps)

        # Centers
        pcx = (pred_boxes[..., 0] + pred_boxes[..., 2]) * 0.5
        pcy = (pred_boxes[..., 1] + pred_boxes[..., 3]) * 0.5
        tcx = (target_boxes[..., 0] + target_boxes[..., 2]) * 0.5
        tcy = (target_boxes[..., 1] + target_boxes[..., 3]) * 0.5

        rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
        _, _, _, _, _, _, c2 = _enclosing_box(pred_boxes, target_boxes, eps=eps)

        # Box dimensions
        pw = (pred_boxes[..., 2] - pred_boxes[..., 0]).clamp(min=eps)
        ph = (pred_boxes[..., 3] - pred_boxes[..., 1]).clamp(min=eps)
        tw = (target_boxes[..., 2] - target_boxes[..., 0]).clamp(min=eps)
        th = (target_boxes[..., 3] - target_boxes[..., 1]).clamp(min=eps)

        # Size terms normalised by max(pred, target) dimension
        w_term = (pw - tw) ** 2 / (torch.max(pw, tw) ** 2 + eps)
        h_term = (ph - th) ** 2 / (torch.max(ph, th) ** 2 + eps)

        loss = 1.0 - iou + rho2 / (c2 + eps) + w_term + h_term
        return _apply_reduction(loss, self.reduction)


class SIoULoss(nn.Module):
    """
    Shape-aware IoU loss — Gevorgyan, arXiv 2022 (https://arxiv.org/abs/2205.12740).

    L = 1 − IoU + (Λ + Ω) / 2

    where:
        Λ (distance cost) = 2 − exp(γ·ρ_x) − exp(γ·ρ_y)
            • ρ_x, ρ_y: normalized center-axis distances (relative to enclosing box dims)
            • γ = angle_cost − 2, encoding how mis-aligned the predicted box is
        Ω (shape cost) = Σ (1 − exp(−ω_d))^θ  for d ∈ {w, h}
            • ω_d = |pred_d − tgt_d| / max(pred_d, tgt_d)  — normalised dim error
            • θ = 4 (sharpens the penalty near zero error)

    Key idea: the angle between the center-offset vector and the nearest GT axis
    modulates how strongly center distance is penalised. If the prediction is
    already moving toward the target along the dominant axis, the penalty is softer.
    This makes gradient flow smoother and improves convergence direction.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred_boxes, target_boxes):
        eps = self.eps
        iou = _bbox_iou_xyxy(pred_boxes, target_boxes, eps=eps)

        # Centers and absolute differences
        pcx = (pred_boxes[..., 0] + pred_boxes[..., 2]) * 0.5
        pcy = (pred_boxes[..., 1] + pred_boxes[..., 3]) * 0.5
        tcx = (target_boxes[..., 0] + target_boxes[..., 2]) * 0.5
        tcy = (target_boxes[..., 1] + target_boxes[..., 3]) * 0.5

        abs_dx = torch.abs(tcx - pcx)
        abs_dy = torch.abs(tcy - pcy)
        sigma  = torch.sqrt(abs_dx ** 2 + abs_dy ** 2 + eps)  # center distance ρ

        # Angle cost ─────────────────────────────────────────────────────────
        # sin of angle between the center-offset vector and the *nearest* axis.
        # sin_alpha_1 = |Δy|/ρ  (angle with horizontal axis)
        # sin_alpha_2 = |Δx|/ρ  (angle with vertical axis)
        # Pick the smaller angle (angle closest to one of the box axes).
        sin_alpha_1 = abs_dy / sigma
        sin_alpha_2 = abs_dx / sigma
        threshold   = (2.0 ** 0.5) / 2.0   # sin(45°) ≈ 0.7071
        sin_alpha   = torch.where(sin_alpha_1 <= threshold, sin_alpha_1, sin_alpha_2)

        # angle_cost ∈ [0, 1]:  0 when perfectly axis-aligned, 1 at 45° offset
        # Using: cos(arcsin(x)·2 − π/2) = sin(2·arcsin(x)) = 2·x·√(1−x²)
        cos_alpha   = torch.sqrt((1.0 - sin_alpha ** 2).clamp(min=eps))
        angle_cost  = torch.cos(torch.asin(sin_alpha) * 2.0 - math.pi / 2.0)

        # Distance cost ───────────────────────────────────────────────────────
        _, _, _, _, cw, ch, _ = _enclosing_box(pred_boxes, target_boxes, eps=eps)
        rho_x = (abs_dx / (cw + eps)) ** 2
        rho_y = (abs_dy / (ch + eps)) ** 2
        gamma  = angle_cost - 2.0   # ∈ [−2, 0]; more negative → faster decay
        distance_cost = 2.0 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)

        # Shape cost ──────────────────────────────────────────────────────────
        pw = (pred_boxes[..., 2] - pred_boxes[..., 0]).clamp(min=eps)
        ph = (pred_boxes[..., 3] - pred_boxes[..., 1]).clamp(min=eps)
        tw = (target_boxes[..., 2] - target_boxes[..., 0]).clamp(min=eps)
        th = (target_boxes[..., 3] - target_boxes[..., 1]).clamp(min=eps)

        omega_w = torch.abs(pw - tw) / (torch.max(pw, tw) + eps)
        omega_h = torch.abs(ph - th) / (torch.max(ph, th) + eps)
        theta   = 4
        shape_cost = (
            torch.pow(1.0 - torch.exp(-omega_w), theta)
            + torch.pow(1.0 - torch.exp(-omega_h), theta)
        )

        loss = 1.0 - iou + (distance_cost + shape_cost) / 2.0
        return _apply_reduction(loss, self.reduction)


class WIoULoss(nn.Module):
    """
    Wise IoU loss v1 — Tong et al., arXiv 2023 (https://arxiv.org/abs/2301.10051).

    L = β · (1 − IoU)

    where:
        β = exp(ρ² / c²)   — wise focusing coefficient

    β up-weights examples whose predicted center is far from the GT center
    (relative to the enclosing-box diagonal). This acts as a dynamic hard-example
    miner: poorly localised predictions receive a larger gradient signal, pulling
    the network away from mediocre local minima that plain IoU loss can settle into.

    Unlike DIoU, WIoU does NOT add ρ²/c² as an additive term; instead it *scales*
    the overlap loss by it. The effect is a loss landscape that naturally de-emphasises
    near-perfect predictions (small ρ → β ≈ 1) and strongly emphasises outliers
    (large ρ → β >> 1).

    Note: This is WIoU v1 (static focusing coefficient). WIoU v3 requires a running
    mean of the loss and is not implemented here for simplicity.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred_boxes, target_boxes):
        eps = self.eps
        iou = _bbox_iou_xyxy(pred_boxes, target_boxes, eps=eps)

        # Center distance ρ²
        pcx = (pred_boxes[..., 0] + pred_boxes[..., 2]) * 0.5
        pcy = (pred_boxes[..., 1] + pred_boxes[..., 3]) * 0.5
        tcx = (target_boxes[..., 0] + target_boxes[..., 2]) * 0.5
        tcy = (target_boxes[..., 1] + target_boxes[..., 3]) * 0.5
        rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

        # Enclosing-box diagonal² (normalisation factor)
        _, _, _, _, _, _, c2 = _enclosing_box(pred_boxes, target_boxes, eps=eps)

        # Wise focusing coefficient — clamp exponent to avoid overflow
        beta = torch.exp((rho2 / (c2 + eps)).clamp(max=10.0))

        loss = beta * (1.0 - iou)
        return _apply_reduction(loss, self.reduction)


class AEIoULoss(nn.Module):
    """
    Amorphous-EIoU (A-EIoU) — proposed loss for objects without rigid boundaries.

    L = (1 − IoU)
        + ρ²/c²                                          [DIoU center-distance term]
        + λ · (w_err²/wt² + h_err²/ht²)                 [size error, target-normalised]

    Key design choices vs EIoU:
    ─────────────────────────────────────────────────────────────────────────────
    1. Normalisation: size errors are divided by the *target* width/height (wt², ht²)
       rather than the enclosing-box dimensions (cw², ch²). This makes the penalty
       proportional to how wrong the prediction is relative to the label — more
       meaningful when labels themselves are noisy/irregular.

    2. λ-rigidity: scales the entire size term. For amorphous objects, ground-truth
       bounding boxes are inherently imprecise (the annotator is guessing the extent
       of an irregular organism). Setting λ < 1 down-weights this noisy signal:
         λ → 0   : AEIoU ≈ DIoU  (only penalise center drift, ignore shape)
         λ = 1.0 : full size penalty active (comparable in strength to EIoU)
       Empirically, λ ≈ 0.3 tends to work best for the DUO dataset.

    3. No aspect-ratio term (v): CIoU's v term enforces w/h ratio consistency.
       Amorphous objects violate this assumption — a scallop can appear as any
       aspect ratio depending on orientation. Removing v avoids penalising valid
       predictions that happen to have a different aspect ratio from the label.

    Parameters
    ──────────
    rigidity : float
        λ in [0, 1]. Controls how strongly shape mismatch is penalised.
        0 = treat all shapes as equally valid (pure center-alignment loss).
        1 = maximum shape penalty (EIoU-equivalent strength).
    reduction : str
        "mean" | "sum" | "none"
    eps : float
        Numerical stability constant.
    """
    def __init__(self, rigidity: float = 1.0, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        self.rigidity = rigidity
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred_boxes, target_boxes):
        """
        pred_boxes, target_boxes: (..., 4) xyxy
        """
        eps = self.eps

        # IoU term
        iou = _bbox_iou_xyxy(pred_boxes, target_boxes, eps=eps)  # (...,)
        one_minus_iou = 1.0 - iou

        # Widths, heights, centers
        px1, py1, px2, py2 = pred_boxes.unbind(-1)
        tx1, ty1, tx2, ty2 = target_boxes.unbind(-1)

        pw = (px2 - px1).clamp(min=eps)
        ph = (py2 - py1).clamp(min=eps)
        tw = (tx2 - tx1).clamp(min=eps)
        th = (ty2 - ty1).clamp(min=eps)

        pcx = (px1 + px2) * 0.5
        pcy = (py1 + py2) * 0.5
        tcx = (tx1 + tx2) * 0.5
        tcy = (ty1 + ty2) * 0.5

        # Smallest enclosing box diag^2 (DIoU-style normalization)
        cx1 = torch.min(px1, tx1)
        cy1 = torch.min(py1, ty1)
        cx2 = torch.max(px2, tx2)
        cy2 = torch.max(py2, ty2)
        cw = (cx2 - cx1).clamp(min=eps)
        ch = (cy2 - cy1).clamp(min=eps)
        c2 = cw * cw + ch * ch  # diag^2

        center_dist2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
        diou_term = center_dist2 / (c2 + eps)

        # Width/height relative error, scaled by λ (rigidity)
        # Normalised by target dims — penalises prediction error proportional to label size
        w_err2 = (pw - tw) ** 2 / (tw * tw + eps)
        h_err2 = (ph - th) ** 2 / (th * th + eps)
        size_term = self.rigidity * (w_err2 + h_err2)

        loss = one_minus_iou + diou_term + size_term  # (...,)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # "none"
