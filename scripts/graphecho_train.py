#!/usr/bin/env python
# ================================================================
# Clean Supervised Baseline — ECHO Project
# ================================================================

import os
import sys
import time
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ================================================================
# Fix Python Path
# ================================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.datasets.cardiac_uda import CardiacUDADataset
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

DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "cardiacUDC_dataset")
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

NUM_CLASSES = 5
BASE_CH = 64
IMG_SIZE = 384

BATCH_SIZE = 2
NUM_EPOCHS = 30
LR = 3e-4
WEIGHT_DECAY = 1e-4

CLASS_WEIGHTS = [0.15, 1.0, 1.0, 1.0, 1.0]

print(f"\nDataset root: {DATA_ROOT}")

# ================================================================
# Dataset
# ================================================================

print("\nDiscovering dataset...")
for item in sorted(os.listdir(DATA_ROOT)):
    full = os.path.join(DATA_ROOT, item)
    if os.path.isdir(full):
        n = len(glob.glob(os.path.join(full, "*_image.nii.gz")))
        print(f"  {item} — {n} volumes")

src_aug = CardiacUDADataset(
    DATA_ROOT,
    domain="G",
    resize=IMG_SIZE,
    augment=True,
    normalize_mode="zscore",
)

src_clean = CardiacUDADataset(
    DATA_ROOT,
    domain="G",
    resize=IMG_SIZE,
    augment=False,
    normalize_mode="zscore",
)

train_size = int(0.8 * len(src_aug))
val_size = len(src_aug) - train_size

train_ds, _ = random_split(
    src_aug,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)

_, val_ds = random_split(
    src_clean,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)

tgt_ds = CardiacUDADataset(
    DATA_ROOT,
    domain="R",
    resize=IMG_SIZE,
    augment=False,
    normalize_mode="zscore",
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
tgt_loader = DataLoader(tgt_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"\nTrain: {len(train_ds)} | Val: {len(val_ds)} | Target: {len(tgt_ds)}")

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
)

loss_fn = DeepSupervisionLoss(base_loss, aux_weights=(0.4, 0.2))

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS,
    eta_min=1e-6,
)

print("Loss + Optimizer ready.")

# ================================================================
# Training Loop
# ================================================================

best_val = 0.0
best_tgt = 0.0

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

    # ---- Validation (Source)
    model.eval()
    val_dice = 0.0
    n_val = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            val_dice += dice_score(logits, masks, NUM_CLASSES)
            n_val += 1

    val_dice /= n_val

    # ---- Validation (Target)
    tgt_dice = 0.0
    n_tgt = 0

    with torch.no_grad():
        for imgs, masks in tgt_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            tgt_dice += dice_score(logits, masks, NUM_CLASSES)
            n_tgt += 1

    tgt_dice /= n_tgt

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

    if val_dice > best_val:
        best_val = val_dice
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best_val.pth"))

    if tgt_dice > best_tgt:
        best_tgt = tgt_dice
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, "best_target.pth"))

print("\nTraining complete.")
print(f"Best Val Dice: {best_val:.4f}")
print(f"Best Target Dice: {best_tgt:.4f}")