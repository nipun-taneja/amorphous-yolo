"""
Custom loss functions for amorphous-yolo:
- EIoU / A-EIoU variants to replace or augment YOLO's default CIoU box loss.
"""

import torch
import torch.nn as nn

class EIoULoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_boxes, target_boxes):
        # TODO: implement EIoU here
        raise NotImplementedError("EIoULoss.forward not implemented yet")


class AEIoULoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred_boxes, target_boxes):
        # TODO: implement A-EIoU here
        raise NotImplementedError("AEIoULoss.forward not implemented yet")
