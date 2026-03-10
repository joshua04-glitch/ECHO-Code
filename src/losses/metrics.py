"""
losses/metrics.py — Loss functions and metrics with ignore_index support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX = 255


def dice_score(logits, targets, num_classes, ignore_index=IGNORE_INDEX):
    """Mean Dice score across foreground classes, ignoring unlabeled pixels."""
    if isinstance(logits, tuple):
        logits = logits[0]

    preds = torch.argmax(logits, dim=1)
    valid = targets != ignore_index

    dice_sum = 0.0
    count = 0

    for c in range(1, num_classes):
        pred_c = (preds == c) & valid
        true_c = (targets == c) & valid

        intersection = (pred_c & true_c).sum().float()
        union = pred_c.sum().float() + true_c.sum().float()

        if union > 0:
            dice_sum += (2.0 * intersection / (union + 1e-6)).item()
            count += 1

    return dice_sum / max(count, 1)


def per_class_dice(logits, targets, num_classes, class_names=None, ignore_index=IGNORE_INDEX):
    """Per-class Dice scores, ignoring unlabeled pixels."""
    if isinstance(logits, tuple):
        logits = logits[0]

    preds = torch.argmax(logits, dim=1)
    valid = targets != ignore_index

    if class_names is None:
        class_names = [f"class_{c}" for c in range(num_classes)]

    results = {}
    for c in range(num_classes):
        pred_c = (preds == c) & valid
        true_c = (targets == c) & valid

        intersection = (pred_c & true_c).sum().float()
        union = pred_c.sum().float() + true_c.sum().float()

        if union > 0:
            results[class_names[c]] = (2.0 * intersection / (union + 1e-6)).item()
        else:
            results[class_names[c]] = float("nan")

    return results


class DiceLoss(nn.Module):
    """Soft Dice loss with ignore_index support."""

    def __init__(self, num_classes, smooth=1.0, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        valid = targets != self.ignore_index

        safe_targets = targets.clone()
        safe_targets[~valid] = 0

        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(safe_targets, self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()

        valid_mask = valid.unsqueeze(1).expand_as(probs).float()

        probs = probs * valid_mask
        one_hot = one_hot * valid_mask

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        union = probs.sum(dims) + one_hot.sum(dims)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice[1:].mean()


class BoundaryLoss(nn.Module):
    """Boundary-aware loss with ignore_index support."""

    def __init__(self, ignore_index=IGNORE_INDEX):
        super().__init__()
        self.ignore_index = ignore_index
        # Store as plain tensors — we move them in _edges()
        self._sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        self._sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)

    def _edges(self, mask):
        m = mask.unsqueeze(1).float()
        sx = self._sobel_x.to(m.device)
        sy = self._sobel_y.to(m.device)
        gx = F.conv2d(m, sx, padding=1)
        gy = F.conv2d(m, sy, padding=1)
        return (gx.abs() + gy.abs()).squeeze(1).clamp(0, 1)

    def forward(self, logits, targets):
        valid = (targets != self.ignore_index).float()

        preds = torch.argmax(logits, dim=1).float()

        safe_targets = targets.clone().float()
        safe_targets[targets == self.ignore_index] = 0

        pred_edges = self._edges(preds) * valid
        true_edges = self._edges(safe_targets) * valid

        n_valid = valid.sum().clamp(min=1)
        return F.mse_loss(pred_edges, true_edges, reduction="sum") / n_valid


class CombinedLoss(nn.Module):
    """CE + Dice + Boundary with ignore_index support."""

    def __init__(self, num_classes, ce_weight=1.0, dice_weight=1.0,
                 boundary_weight=0.5, class_weights=None,
                 ignore_index=IGNORE_INDEX):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.ignore_index = ignore_index

        self._class_weights_list = class_weights

        self.dice = DiceLoss(num_classes, ignore_index=ignore_index)
        self.boundary = BoundaryLoss(ignore_index=ignore_index)

    def forward(self, logits, targets):
        if self._class_weights_list is not None:
            w = torch.tensor(self._class_weights_list,
                             dtype=logits.dtype, device=logits.device)
        else:
            w = None

        loss = self.ce_weight * F.cross_entropy(
            logits, targets,
            weight=w,
            ignore_index=self.ignore_index,
        )
        loss += self.dice_weight * self.dice(logits, targets)
        if self.boundary_weight > 0:
            loss += self.boundary_weight * self.boundary(logits, targets)
        return loss


class DeepSupervisionLoss(nn.Module):
    """Deep supervision wrapper."""

    def __init__(self, base_loss, aux_weights=(0.4, 0.2)):
        super().__init__()
        self.base_loss = base_loss
        self.aux_weights = aux_weights

    def forward(self, outputs, targets):
        if isinstance(outputs, tuple):
            main_logits, aux_list = outputs
            loss = self.base_loss(main_logits, targets)
            for i, aux in enumerate(aux_list):
                if i < len(self.aux_weights):
                    loss += self.aux_weights[i] * self.base_loss(aux, targets)
            return loss
        else:
            return self.base_loss(outputs, targets)