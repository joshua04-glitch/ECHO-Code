#!/usr/bin/env python
"""
Cardiac segmentation training v2 — all 4 chambers, improved.

Data sources:
  - CardiacUDA G (full labels: LV, RV, RA, LA) — oversampled 4×
  - CAMUS (partial: LV, LA — no RA, no RV)
  - PLOSONE (all 4 chambers: LV, LA, RA, RV)

Improvements over v1:
  1. CardiacUDA-G oversampled 4× for balanced full-label exposure
  2. Batch size 4 for more stable gradients
  3. CosineAnnealingWarmRestarts scheduler
  4. Test-Time Augmentation (TTA) for evaluation
  5. 50 epochs with early stopping
  6. Slight RV class weight boost
"""

import os
import sys
import time
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, ConcatDataset
from tqdm import tqdm

DATAROOT = "/scratch/users/joshua04/ECHO"
PROJECT_ROOT = "/scratch/users/joshua04/Echo-Code"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.datasets.cardiac_uda import CardiacUDADataset, PLOSONEDataset
from src.models.unet import UNet
from src.losses.metrics import (
    dice_score,
    per_class_dice,
    CombinedLoss,
    DeepSupervisionLoss,
)

# ================================================================
# Device
# ================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ================================================================
# Config
# ================================================================

DATA_ROOT = os.path.join(DATAROOT, "data", "cardiacUDC_dataset")
PLOSONE_ROOT = os.path.join(DATAROOT, "data", "PLOSONE")
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

NUM_CLASSES = 5
CLASS_NAMES = ["BG", "LV", "LA", "RA", "RV"]
BASE_CH = 64
IMG_SIZE = 384

BATCH_SIZE = 4
NUM_EPOCHS = 50
LR = 3e-4
WEIGHT_DECAY = 1e-4

CLASS_WEIGHTS = [0.15, 1.0, 1.0, 1.0, 1.2]

G_OVERSAMPLE = 4

print(f"\nCardiacUDA root: {DATA_ROOT}")
print(f"PLOSONE root:    {PLOSONE_ROOT}")

# ================================================================
# Datasets
# ================================================================

print("\nDiscovering dataset...")
for item in sorted(os.listdir(DATA_ROOT)):
    full = os.path.join(DATA_ROOT, item)
    if os.path.isdir(full):
        n = len(glob.glob(os.path.join(full, "*_image.nii.gz")))
        if n > 0 and not item.startswith("Site_RVENet"):
            print(f"  {item} -- {n} volumes")

# ── CardiacUDA-G (full labels) — oversampled ──
src_aug_list = []
src_clean_list = []
for _ in range(G_OVERSAMPLE):
    src_aug_list.append(CardiacUDADataset(
        DATA_ROOT, domain="G", resize=IMG_SIZE,
        augment=True, normalize_mode="zscore",
    ))
    src_clean_list.append(CardiacUDADataset(
        DATA_ROOT, domain="G", resize=IMG_SIZE,
        augment=False, normalize_mode="zscore",
    ))

# ── CAMUS (partial: LV + LA only) ──
camus_aug = CardiacUDADataset(
    DATA_ROOT, domain="CAMUS", resize=IMG_SIZE,
    augment=True, normalize_mode="zscore",
)
camus_clean = CardiacUDADataset(
    DATA_ROOT, domain="CAMUS", resize=IMG_SIZE,
    augment=False, normalize_mode="zscore",
)

# ── RVENet — SKIP: labels are all zeros (unannotated) ──
print("RVENet: skipped (labels are empty/unannotated)")

# ── PLOSONE (all 4 chambers) ──
plosone_aug = PLOSONEDataset(
    PLOSONE_ROOT, resize=IMG_SIZE, augment=True,
)
plosone_clean = PLOSONEDataset(
    PLOSONE_ROOT, resize=IMG_SIZE, augment=False,
)

# ── Combine all sources ──
full_aug = ConcatDataset(src_aug_list + [camus_aug, plosone_aug])
full_clean = ConcatDataset(src_clean_list + [camus_clean, plosone_clean])

train_size = int(0.8 * len(full_aug))
val_size = len(full_aug) - train_size

train_ds, _ = random_split(
    full_aug,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)
_, val_ds = random_split(
    full_clean,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)

# Target domain (R)
tgt_ds = CardiacUDADataset(
    DATA_ROOT, domain="R", resize=IMG_SIZE,
    augment=False, normalize_mode="zscore",
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)
tgt_loader   = DataLoader(tgt_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)

print(f"\nTotal combined: {len(full_aug)}")
print(f"  G (x{G_OVERSAMPLE}): {sum(len(d) for d in src_aug_list)}")
print(f"  CAMUS: {len(camus_aug)}")
print(f"  PLOSONE: {len(plosone_aug)}")
print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Target: {len(tgt_ds)}")

# ================================================================
# Model
# ================================================================

model = UNet(
    in_ch=1,
    num_classes=NUM_CLASSES,
    base_ch=BASE_CH,
    use_attention=True,
    deep_supervision=True,
    dropout=0.15,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

# ================================================================
# Loss + Optimizer
# ================================================================

base_loss = CombinedLoss(
    num_classes=NUM_CLASSES,
    ce_weight=1.0,
    dice_weight=1.0,
    boundary_weight=0.5,
    class_weights=CLASS_WEIGHTS,
    ignore_index=255,
)

loss_fn = DeepSupervisionLoss(base_loss, aux_weights=(0.4, 0.2))

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=15, T_mult=2, eta_min=1e-6,
)

print("Loss + Optimizer ready.")

# ================================================================
# TTA (Test-Time Augmentation)
# ================================================================

def predict_with_tta(model, imgs):
    """Average predictions over original + horizontally flipped."""
    logits1 = model(imgs)

    imgs_flip = torch.flip(imgs, dims=[3])
    logits2 = model(imgs_flip)
    logits2 = torch.flip(logits2, dims=[3])

    return (logits1 + logits2) / 2.0

# ================================================================
# Training Loop
# ================================================================

best_val = 0.0
best_tgt = 0.0
patience_counter = 0
PATIENCE = 15

print("\nStarting training...\n")

for epoch in range(NUM_EPOCHS):
    t0 = time.time()
    model.train()

    train_loss = 0.0
    train_dice = 0.0
    n_train = 0

    for imgs, masks in tqdm(train_loader, leave=False):
        imgs = imgs.to(device)
        masks = masks.to(device)

        outputs = model(imgs)
        loss = loss_fn(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        with torch.no_grad():
            main = outputs[0] if isinstance(outputs, tuple) else outputs
            train_dice += dice_score(main, masks, NUM_CLASSES)

        train_loss += loss.item()
        n_train += 1

    scheduler.step()

    train_loss /= n_train
    train_dice /= n_train

    # ---- Validation (Source) with TTA ----
    model.eval()
    val_dice = 0.0
    val_class_dice = {c: 0.0 for c in CLASS_NAMES}
    n_val = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = predict_with_tta(model, imgs)
            val_dice += dice_score(logits, masks, NUM_CLASSES)

            cd = per_class_dice(logits, masks, NUM_CLASSES, CLASS_NAMES)
            for c in CLASS_NAMES:
                if not np.isnan(cd[c]):
                    val_class_dice[c] += cd[c]
            n_val += 1

    val_dice /= n_val
    for c in CLASS_NAMES:
        val_class_dice[c] /= n_val

    # ---- Validation (Target) with TTA ----
    tgt_dice = 0.0
    tgt_class_dice = {c: 0.0 for c in CLASS_NAMES}
    n_tgt = 0

    with torch.no_grad():
        for imgs, masks in tgt_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = predict_with_tta(model, imgs)
            tgt_dice += dice_score(logits, masks, NUM_CLASSES)

            cd = per_class_dice(logits, masks, NUM_CLASSES, CLASS_NAMES)
            for c in CLASS_NAMES:
                if not np.isnan(cd[c]):
                    tgt_class_dice[c] += cd[c]
            n_tgt += 1

    tgt_dice /= n_tgt
    for c in CLASS_NAMES:
        tgt_class_dice[c] /= n_tgt

    elapsed = time.time() - t0
    lr_now = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
        f"Loss {train_loss:.4f} | "
        f"TrainDice {train_dice:.4f} | "
        f"Val {val_dice:.4f} | "
        f"Tgt {tgt_dice:.4f} | "
        f"LR {lr_now:.6f} | "
        f"{elapsed:.0f}s"
    )

    tgt_str = " | ".join(f"{c}:{tgt_class_dice[c]:.3f}" for c in CLASS_NAMES[1:])
    val_str = " | ".join(f"{c}:{val_class_dice[c]:.3f}" for c in CLASS_NAMES[1:])
    print(f"  Val  per-class: {val_str}")
    print(f"  Tgt  per-class: {tgt_str}")

    improved = False
    if val_dice > best_val:
        best_val = val_dice
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best_val_v2.pth"))
        print(f"  -> Saved best val model (Dice {val_dice:.4f})")
        improved = True

    if tgt_dice > best_tgt:
        best_tgt = tgt_dice
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best_target_v2.pth"))
        print(f"  -> Saved best target model (Dice {tgt_dice:.4f})")
        improved = True

    if improved:
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
            break

print("\nTraining complete.")
print(f"Best Val Dice: {best_val:.4f}")
print(f"Best Target Dice: {best_tgt:.4f}")