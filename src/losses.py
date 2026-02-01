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
    Supports reduction='none' for per-box losses (needed for YOLO weighting),
    as well as 'mean' and 'sum'.
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
        # "none" or any other string: return elementwise loss
        return loss


class AEIoULoss(nn.Module):
    """
    Another placeholder; here we just square the (1 - IoU) term.
    Also supports reduction='none' for consistency.
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_boxes, target_boxes):
        iou = _bbox_iou_xyxy(pred_boxes, target_boxes)  # shape: (...,)
        loss = (1.0 - iou) ** 2

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        # "none" or any other string: return elementwise loss
        return loss
