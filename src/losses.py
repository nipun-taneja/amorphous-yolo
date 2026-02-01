import torch
import torch.nn as nn

def _bbox_iou_xyxy(pred_boxes, target_boxes, eps=1e-7):
    """
    Basic IoU in xyxy format, just to make the loss callable.
    pred_boxes, target_boxes: (..., 4) with [x1, y1, x2, y2].
    """
    x1 = torch.max(pred_boxes[..., 0], target_boxes[..., 0])
    y1 = torch.max(pred_boxes[..., 1], target_boxes[..., 1])
    x2 = torch.min(pred_boxes[..., 2], target_boxes[..., 2])
    y2 = torch.min(pred_boxes[..., 3], target_boxes[..., 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area_p = (pred_boxes[..., 2] - pred_boxes[..., 0]).clamp(min=0) * \
             (pred_boxes[..., 3] - pred_boxes[..., 1]).clamp(min=0)
    area_t = (target_boxes[..., 2] - target_boxes[..., 0]).clamp(min=0) * \
             (target_boxes[..., 3] - target_boxes[..., 1]).clamp(min=0)

    union = area_p + area_t - inter + eps
    return inter / union


class EIoULoss(nn.Module):
    """
    Placeholder: right now this behaves like (1 - IoU).
    Later you'll extend it to full EIoU / A-EIoU math.
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_boxes, target_boxes):
        iou = _bbox_iou_xyxy(pred_boxes, target_boxes)
        loss = 1.0 - iou  # placeholder
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class AEIoULoss(nn.Module):
    """
    Another placeholder variant; you can differentiate later.
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_boxes, target_boxes):
        iou = _bbox_iou_xyxy(pred_boxes, target_boxes)
        loss = (1.0 - iou) ** 2  # example variant
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
