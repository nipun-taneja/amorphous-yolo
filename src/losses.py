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


class EIoULoss(nn.Module):
    """
    Placeholder version: behaves like (1 - IoU) loss in xyxy.
    Supports reduction='none' for YOLO weighting, plus 'mean' and 'sum'.
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_boxes, target_boxes):
        iou = _bbox_iou_xyxy(pred_boxes, target_boxes)  # shape: (...,)
        loss = 1.0 - iou

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class AEIoULoss(nn.Module):
    """
    Amorphous-EIoU (A-EIoU) with λ-rigidity.

    Loss:
        L = 1 - IoU
            + center_dist^2 / diag^2      (DIoU-style term)
            + λ * ( w_err^2 / (w_t^2 + eps) + h_err^2 / (h_t^2 + eps) )

    Where boxes are in xyxy format.
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
        w_err2 = (pw - tw) ** 2 / (tw * tw + eps)
        h_err2 = (ph - th) ** 2 / (th * th + eps)
        size_term = self.rigidity * (w_err2 + h_err2)

        loss = one_minus_iou + diou_term + size_term  # (...,)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # "none"
