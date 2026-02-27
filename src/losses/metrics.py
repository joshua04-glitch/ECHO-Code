"""
src/losses/metrics.py — Improved Losses and Metrics
===================================================

Critical fix: dice_score EXCLUDES background by default.

Includes:
  - SoftDiceLoss (foreground only)
  - BoundaryLoss
  - CombinedLoss (GPU-safe class weights)
  - DeepSupervisionLoss
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================
# METRICS (non-differentiable — for evaluation only)
# ====================================================================

def dice_score(logits, targets, num_classes=5, include_background=False, eps=1e-6):
    """
    Mean Dice score over present classes.
    Background excluded by default.
    """
    preds = torch.argmax(logits, dim=1)
    start = 0 if include_background else 1

    scores = []
    for c in range(start, num_classes):
        pred_c = (preds == c).float()
        tgt_c = (targets == c).float()
        inter = (pred_c * tgt_c).sum()
        union = pred_c.sum() + tgt_c.sum()

        if union < eps:
            continue
        scores.append(((2.0 * inter + eps) / (union + eps)).item())

    return sum(scores) / len(scores) if scores else 0.0


def per_class_dice(logits, targets, num_classes=5, eps=1e-6):
    """
    Per-class Dice. Returns dict {class_id: dice}.
    """
    preds = torch.argmax(logits, dim=1)
    out = {}
    for c in range(num_classes):
        pred_c = (preds == c).float()
        tgt_c = (targets == c).float()
        inter = (pred_c * tgt_c).sum()
        union = pred_c.sum() + tgt_c.sum()
        out[c] = (
            ((2.0 * inter + eps) / (union + eps)).item()
            if union > eps
            else float("nan")
        )
    return out


# ====================================================================
# LOSSES
# ====================================================================

class SoftDiceLoss(nn.Module):
    """
    Foreground-only Soft Dice.
    """

    def __init__(self, num_classes=5, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        tgt_oh = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        dice_sum = 0.0
        for c in range(1, self.num_classes):  # skip background
            p = probs[:, c]
            t = tgt_oh[:, c]
            inter = (p * t).sum()
            union = p.sum() + t.sum()
            dice_sum += (2.0 * inter + self.smooth) / (union + self.smooth)

        return 1.0 - dice_sum / (self.num_classes - 1)


class BoundaryLoss(nn.Module):
    """
    Boundary-weighted Cross Entropy.
    """

    def __init__(self, boundary_weight=3.0, kernel_size=3):
        super().__init__()
        self.bw = boundary_weight
        self.ks = kernel_size
        self.pad = kernel_size // 2

    def _boundary_mask(self, targets):
        B, H, W = targets.shape
        boundary = torch.zeros(B, H, W, device=targets.device)

        for c in range(1, 5):
            m = (targets == c).float().unsqueeze(1)
            dilated = F.max_pool2d(m, self.ks, stride=1, padding=self.pad)
            eroded = 1.0 - F.max_pool2d(1.0 - m, self.ks, stride=1, padding=self.pad)
            boundary = torch.max(boundary, (dilated - eroded).squeeze(1))

        return boundary

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        bdry = self._boundary_mask(targets)
        weight = 1.0 + bdry * (self.bw - 1.0)
        return (ce * weight).mean()


class CombinedLoss(nn.Module):
    """
    CE + Dice + Boundary (GPU-safe).
    """

    def __init__(
        self,
        num_classes=5,
        ce_weight=1.0,
        dice_weight=1.0,
        boundary_weight=0.0,
        class_weights=None,
    ):
        super().__init__()

        self.w_ce = ce_weight
        self.w_dice = dice_weight
        self.w_bdry = boundary_weight
        self.num_classes = num_classes

        # 🔥 GPU-SAFE CLASS WEIGHTS FIX
        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
            assert len(w) == num_classes
            self.register_buffer("ce_weight_tensor", w)
            self.ce = nn.CrossEntropyLoss(weight=self.ce_weight_tensor)
        else:
            self.ce = nn.CrossEntropyLoss()

        self.dice = SoftDiceLoss(num_classes)
        self.boundary = BoundaryLoss() if boundary_weight > 0 else None

    def forward(self, logits, targets):
        C = logits.shape[1]
        assert C == self.num_classes

        loss = torch.tensor(0.0, device=logits.device)

        if self.w_ce > 0:
            loss = loss + self.w_ce * self.ce(logits, targets)

        if self.w_dice > 0:
            loss = loss + self.w_dice * self.dice(logits, targets)

        if self.w_bdry > 0 and self.boundary is not None:
            loss = loss + self.w_bdry * self.boundary(logits, targets)

        return loss


class DeepSupervisionLoss(nn.Module):
    """
    Wrap base loss for deep supervision outputs.
    """

    def __init__(self, base_loss, aux_weights=(0.4, 0.2)):
        super().__init__()
        self.base_loss = base_loss
        self.aux_weights = aux_weights

    def forward(self, output, targets):
        if isinstance(output, tuple) and len(output) == 2:
            main, aux_list = output
        else:
            return self.base_loss(output, targets)

        loss = self.base_loss(main, targets)

        for aux, w in zip(aux_list, self.aux_weights):
            loss = loss + w * self.base_loss(aux, targets)

        return loss