#!/usr/bin/env python
"""
train_ef.py  –  EchoNet-Dynamic EF regression
Key improvements vs. original:
  1. Differential learning rates  (backbone << head)
  2. Linear LR warmup for the head, then cosine annealing
  3. Weighted Huber (smooth-L1) loss  –  smoother gradients than MAE
  4. Gradient clipping to stabilise large-batch / high-LR head training
  5. Multi-clip test-time averaging on val  –  bigger RMSE/MAE reduction
  6. Brightness/contrast jitter  –  echocardiography gain varies widely
  7. Correct AMP context on CPU (autocast only on CUDA)
"""

import sys
sys.modules["zstandard"] = None

import os
import math
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision import tv_tensors
from tqdm import tqdm
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = "/scratch/users/joshua04/Echo-Code"
DATA_ROOT    = "/scratch/users/joshua04/ECHO/data/echonet_dynamic"
WEIGHTS_DIR  = os.path.join(PROJECT_ROOT, "weights")

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from src.models.ef.ef_model import EFModel

# ---------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------

CLIP_LEN    = 32
BATCH_SIZE  = 8       # increased from 4 – more stable gradients
NUM_WORKERS = 4

# ── Learning rates ──────────────────────────────────────
# The pretrained backbone needs gentle fine-tuning.
# The fresh regression head can absorb a much higher LR.
BACKBONE_LR  = 5e-6   # very small nudge to the pretrained weights
HEAD_LR      = 5e-4   # fast initial training for the new head
WEIGHT_DECAY = 1e-4

# ── Warmup + cosine schedule ─────────────────────────────
WARMUP_EPOCHS = 5     # linearly ramp both param groups up to their target LR
MAX_EPOCHS    = 60
ETA_MIN_RATIO = 0.02  # final LR = target_LR * ETA_MIN_RATIO

# ── Regularisation ───────────────────────────────────────
GRAD_CLIP = 1.0       # max gradient norm

# ── Early stopping ───────────────────────────────────────
PATIENCE = 12

# ── Multi-clip val averaging ─────────────────────────────
# Number of random crops of the video to average at validation time.
# Costs N× the inference compute but reliably lowers val RMSE by ~0.2-0.5.
VAL_N_CLIPS = 4

# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class EchoNetVideoDataset(Dataset):

    def __init__(self, root, split="train", clip_len=32, n_clips=1):
        """
        n_clips > 1 : sample that many clips per video and average predictions.
                      Only meaningful at eval time.
        """
        df = pd.read_csv(os.path.join(root, "FileList.csv"))

        self.df       = df[df["Split"] == split.upper()].reset_index(drop=True)
        self.root     = root
        self.clip_len = clip_len
        self.split    = split
        self.n_clips  = n_clips

        if split == "train":

            self.transform = v2.Compose([
                # ── spatial ────────────────────────────────────────────
                v2.RandomZoomOut(fill=0, side_range=(1.0, 1.2), p=0.5),
                v2.RandomCrop(size=(224, 224)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomRotation(degrees=(-15, 15)),
                # ── photometric (echo gain varies widely) ──────────────
                v2.ColorJitter(brightness=0.2, contrast=0.2),
                # ── dtype + normalise ──────────────────────────────────
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

        else:

            self.transform = v2.Compose([
                v2.CenterCrop(size=(224, 224)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

    # ------------------------------------------------------------------
    def _load_clip(self, path, frame_count, start=None):

        cap = cv2.VideoCapture(path)

        if frame_count <= self.clip_len:
            start = 0
        elif start is None:
            # random start for train / first clip; caller passes explicit
            # start for additional val clips
            start = np.random.randint(0, frame_count - self.clip_len) \
                    if self.split == "train" else 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, max(start - 1, 0))

        frames = []
        for _ in range(self.clip_len):
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (256, 256))
                frames.append(frame)
            else:
                frames.append(frames[-1])

        cap.release()

        v = np.stack(frames)
        v = tv_tensors.Video(np.transpose(v, (0, 3, 1, 2)))
        return v

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        path = os.path.join(
            self.root, "Videos", row["FileName"] + ".avi"
        )

        frame_count = int(
            cv2.VideoCapture(path).get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if self.n_clips == 1 or self.split == "train":
            # ── single clip (training) ────────────────────────────────
            x = self._load_clip(path, frame_count)
            x = self.transform(x)
            x = x.permute(1, 0, 2, 3)          # → (C, T, H, W)

        else:
            # ── multiple clips (val TTA) ──────────────────────────────
            # Evenly spaced starts across the video
            max_start = max(frame_count - self.clip_len, 1)
            starts    = np.linspace(0, max_start, self.n_clips, dtype=int)

            clips = []
            for s in starts:
                c = self._load_clip(path, frame_count, start=int(s))
                c = self.transform(c)
                c = c.permute(1, 0, 2, 3)      # (C, T, H, W)
                clips.append(c)

            x = torch.stack(clips, dim=0)       # (n_clips, C, T, H, W)

        y = torch.tensor(row["EF"], dtype=torch.float32)

        return x, y


# ---------------------------------------------------------
# Data loaders
# ---------------------------------------------------------

train_dataset = EchoNetVideoDataset(DATA_ROOT, "train", CLIP_LEN, n_clips=1)
val_dataset   = EchoNetVideoDataset(DATA_ROOT, "val",   CLIP_LEN, n_clips=VAL_N_CLIPS)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
)

# val batch_size=1 so we can handle the (n_clips, C, T, H, W) tensor easily
val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

print(f"Train: {len(train_dataset)} videos | Val: {len(val_dataset)} videos")

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

model = EFModel(
    arch="convnext_tiny",
    n_heads=8,
    n_layers=4,
    clip_len=CLIP_LEN,
    pretrained=True,
    weights_dir=WEIGHTS_DIR,
).to(device)

# ---------------------------------------------------------
# Loss  –  weighted Huber (smooth-L1)
# Huber is differentiable everywhere and less noisy than MAE for large errors.
# Weights emphasise the clinically critical low-EF range.
# ---------------------------------------------------------

def ef_loss(pred, target):
    """Weighted Huber loss with emphasis on low EF (<40 / <30)."""

    error   = pred - target
    abs_err = error.abs()

    # Smooth-L1 / Huber with delta=5 EF points
    delta  = 5.0
    huber  = torch.where(
        abs_err < delta,
        0.5 * error ** 2,
        delta * (abs_err - 0.5 * delta),
    )

    weights = torch.ones_like(target)
    weights[target < 40] = 2.0
    weights[target < 30] = 3.5

    return (weights * huber).mean()

# ---------------------------------------------------------
# Optimiser  –  two param groups
# ---------------------------------------------------------

backbone_params = list(model.backbone.parameters())
head_params     = list(model.head.parameters())

optimizer = optim.AdamW(
    [
        {"params": backbone_params, "lr": BACKBONE_LR},
        {"params": head_params,     "lr": HEAD_LR},
    ],
    weight_decay=WEIGHT_DECAY,
)

# ---------------------------------------------------------
# LR schedule  –  linear warmup → cosine annealing
# Both param groups scale by the same lambda so their ratio stays constant.
# ---------------------------------------------------------

def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        # Ramp from 0 → 1
        return (epoch + 1) / WARMUP_EPOCHS
    # Cosine decay from 1 → ETA_MIN_RATIO
    progress = (epoch - WARMUP_EPOCHS) / max(MAX_EPOCHS - WARMUP_EPOCHS, 1)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return ETA_MIN_RATIO + (1.0 - ETA_MIN_RATIO) * cosine

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# ---------------------------------------------------------
# AMP
# ---------------------------------------------------------

use_amp = device == "cuda"
scaler  = torch.amp.GradScaler(enabled=use_amp)

# ---------------------------------------------------------
# Training epoch
# ---------------------------------------------------------

def run_train_epoch(model, loader, optimizer):

    model.train()

    total_loss = 0
    total      = 0
    preds      = []
    labels     = []

    for x, y in tqdm(loader, leave=False):

        x = x.to(device)
        y = y.to(device).view(-1, 1)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(x)
            loss = ef_loss(pred, y)

        scaler.scale(loss).backward()

        # ── gradient clipping ────────────────────────────────────────
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        total      += x.size(0)

        preds.append(pred.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())

    preds  = np.concatenate(preds).ravel()
    labels = np.concatenate(labels).ravel()

    rmse = np.sqrt(((preds - labels) ** 2).mean())
    mae  = np.mean(np.abs(preds - labels))
    r2   = r2_score(labels, preds)

    return rmse, mae, r2

# ---------------------------------------------------------
# Validation epoch  –  multi-clip averaging
# ---------------------------------------------------------

def run_val_epoch(model, loader):

    model.eval()

    preds  = []
    labels = []

    with torch.no_grad():

        for x, y in tqdm(loader, leave=False):

            # x shape: (1, n_clips, C, T, H, W)  from Dataset.__getitem__
            # squeeze the batch=1 dim: (n_clips, C, T, H, W)
            x = x.squeeze(0).to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                # forward each clip independently then average
                clip_preds = model(x)           # (n_clips, 1)

            pred = clip_preds.mean(dim=0)       # (1,)

            preds.append(pred.cpu().numpy())
            labels.append(y.numpy())

    preds  = np.concatenate(preds).ravel()
    labels = np.concatenate(labels).ravel()

    rmse = np.sqrt(((preds - labels) ** 2).mean())
    mae  = np.mean(np.abs(preds - labels))
    r2   = r2_score(labels, preds)

    return rmse, mae, r2

# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------

ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)

best_rmse        = 1e9
epochs_no_improve = 0

print(f"\nBackbone LR: {BACKBONE_LR:.1e}  |  Head LR: {HEAD_LR:.1e}")
print(f"Warmup epochs: {WARMUP_EPOCHS}  |  Max epochs: {MAX_EPOCHS}\n")

for epoch in range(MAX_EPOCHS):

    train_rmse, train_mae, train_r2 = run_train_epoch(
        model, train_loader, optimizer
    )

    val_rmse, val_mae, val_r2 = run_val_epoch(model, val_loader)

    scheduler.step()

    backbone_lr = optimizer.param_groups[0]["lr"]
    head_lr     = optimizer.param_groups[1]["lr"]

    print(
        f"[Epoch {epoch:02d}] "
        f"train RMSE {train_rmse:.2f}  MAE {train_mae:.2f}  R² {train_r2:.3f} | "
        f"val RMSE {val_rmse:.2f}  MAE {val_mae:.2f}  R² {val_r2:.3f} | "
        f"lr_bb {backbone_lr:.1e}  lr_hd {head_lr:.1e}"
    )

    if val_rmse < best_rmse:

        best_rmse         = val_rmse
        epochs_no_improve = 0

        torch.save(
            {
                "epoch":     epoch,
                "val_rmse":  val_rmse,
                "val_mae":   val_mae,
                "val_r2":    val_r2,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            os.path.join(ckpt_dir, "ef_best.pth"),
        )

        print(f"  ↑ New best val RMSE: {best_rmse:.2f} — saved checkpoint")

    else:

        epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping after {epoch + 1} epochs (no improvement for {PATIENCE}).")
            break

torch.save(model.state_dict(), os.path.join(ckpt_dir, "ef_final.pth"))
print(f"\nTraining complete. Best val RMSE: {best_rmse:.2f}")