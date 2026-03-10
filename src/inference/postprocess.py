"""
inference/postprocess.py — Post-processing and test-time augmentation utilities.
"""

import numpy as np
import torch
from scipy import ndimage


def largest_component(mask, num_classes=5):
    """
    Keep only the largest connected component per class.
    Also fills holes inside regions.

    Args:
        mask: (H, W) int array with class labels 0..num_classes-1
        num_classes: total number of classes (including BG)

    Returns:
        cleaned: (H, W) int array
    """
    cleaned = np.zeros_like(mask)

    for c in range(1, num_classes):
        binary = mask == c

        if binary.sum() == 0:
            continue

        binary = ndimage.binary_fill_holes(binary)

        labeled, num = ndimage.label(binary)

        if num == 1:
            cleaned[binary] = c
            continue

        sizes = ndimage.sum(binary, labeled, range(1, num + 1))
        largest = sizes.argmax() + 1
        cleaned[labeled == largest] = c

    return cleaned


def predict_with_tta(model, imgs):
    """
    Test-time augmentation: average original + horizontal flip.

    Args:
        model: segmentation model
        imgs: (B, 1, H, W) tensor

    Returns:
        logits: (B, C, H, W) averaged logits
    """
    logits1 = model(imgs)

    imgs_flip = torch.flip(imgs, dims=[3])
    logits2 = model(imgs_flip)
    logits2 = torch.flip(logits2, dims=[3])

    return (logits1 + logits2) / 2.0