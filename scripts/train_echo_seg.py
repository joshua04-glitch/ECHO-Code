#!/usr/bin/env python
"""
train_echo_seg.py
─────────────────
Trains the existing UNet on EchoNet-Dynamic LV contour tracings, then
evaluates EDV, ESV, Stroke Volume, and EF on the test set.

Everything is in one file. No new model definitions — uses src/models/unet.py.

Usage:
    python scripts/train_echo_seg.py            # train + evaluate
    python scripts/train_echo_seg.py --eval-only # skip training, just evaluate
"""

import sys
sys.modules["zstandard"] = None

import os
import cv2
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.signal import savgol_filter, find_peaks

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

PROJECT_ROOT = "/scratch/users/joshua04/Echo-Code"
DATA_ROOT    = "/scratch/users/joshua04/ECHO/data/echonet_dynamic"
CKPT_OUT     = os.path.join(PROJECT_ROOT, "checkpoints", "echo_seg_best.pth")

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from src.models.unet import UNet

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE    = 384       # must match your UNet's expected input
LV_LABEL    = 1         # label index for LV in the 5-class scheme
NUM_CLASSES = 5         # must match your saved checkpoint
BATCH_SIZE  = 8
NUM_WORKERS = 4
LR          = 3e-4
WEIGHT_DECAY= 1e-5
MAX_EPOCHS  = 40
PATIENCE    = 8

print(f"Device: {DEVICE}")

# ═════════════════════════════════════════════════════════════════════════════
# 1. PARSE VolumeTracings.csv  →  {filename: {frame_idx: contour_points}}
# ═════════════════════════════════════════════════════════════════════════════

def parse_volume_tracings(data_root: str) -> dict:
    """
    VolumeTracings.csv format (EchoNet-Dynamic):
        FileName, Frame, Calc, X1, Y1, X2, Y2, ... Xn, Yn

    Calc is "ED" or "ES".  Each row is one contour tracing.
    Returns: {filename_no_ext: {frame_int: np.array shape (N,2)}}
    """

    path = os.path.join(data_root, "VolumeTracings.csv")
    df   = pd.read_csv(path)

    # Normalise column names
    df.columns = [c.strip() for c in df.columns]

    tracings = {}

    # Format: each row is one segment (X1,Y1) -> (X2,Y2)
    # Group all segments by (FileName, Frame) then collect unique points
    grouped = df.groupby(["FileName", "Frame"])

    for (filename, frame), group in grouped:

        fname = str(filename).replace(".avi", "")
        frame = int(frame)

        # Collect all endpoint coordinates
        xs = list(group["X1"].values) + list(group["X2"].values)
        ys = list(group["Y1"].values) + list(group["Y2"].values)

        pts = np.stack([xs, ys], axis=1).astype(np.float64)

        # Sort points into a polygon using angle from centroid
        cx, cy = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        pts    = pts[np.argsort(angles)]

        if len(pts) < 3:
            continue

        if fname not in tracings:
            tracings[fname] = {}

        tracings[fname][frame] = pts

    print(f"Parsed tracings for {len(tracings)} videos")

    return tracings


def contour_to_mask(pts: np.ndarray, orig_w: int, orig_h: int, out_size: int) -> np.ndarray:
    """
    Scale contour points from original video resolution to out_size×out_size
    and fill a binary mask.  Returns uint8 mask with LV_LABEL where LV is.
    """

    mask = np.zeros((out_size, out_size), dtype=np.uint8)

    # Scale points
    scaled = pts.copy()
    scaled[:, 0] = scaled[:, 0] * out_size / orig_w
    scaled[:, 1] = scaled[:, 1] * out_size / orig_h

    poly = scaled.astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [poly], color=LV_LABEL)

    return mask


# ═════════════════════════════════════════════════════════════════════════════
# 2. DATASET
# ═════════════════════════════════════════════════════════════════════════════

class EchoSegDataset(Dataset):
    """
    Each sample is a (frame, mask) pair where the mask has LV filled in.
    Only annotated frames (ED and ES) are used for training.
    """

    def __init__(self, data_root: str, split: str, tracings: dict, img_size: int = IMG_SIZE):

        file_list = pd.read_csv(os.path.join(data_root, "FileList.csv"))
        split_files = set(
            file_list[file_list["Split"] == split.upper()]["FileName"]
            .str.replace(".avi", "", regex=False)
            .tolist()
        )

        self.samples  = []   # (video_path, frame_idx, contour_pts, orig_w, orig_h)
        self.img_size = img_size
        self.split    = split

        video_dir = os.path.join(data_root, "Videos")

        for fname, frame_dict in tracings.items():

            if fname not in split_files:
                continue

            video_path = os.path.join(video_dir, fname + ".avi")

            if not os.path.exists(video_path):
                continue

            cap   = cv2.VideoCapture(video_path)
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            for frame_idx, pts in frame_dict.items():
                self.samples.append((video_path, frame_idx, pts, orig_w, orig_h))

        print(f"[{split}] {len(self.samples)} annotated frames from {len(split_files)} videos")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        video_path, frame_idx, pts, orig_w, orig_h = self.samples[idx]

        # ── load frame ────────────────────────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            frame = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

        frame = cv2.resize(frame, (self.img_size, self.img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # ── augmentation (train only) ─────────────────────────────────────
        if self.split == "train":

            # Horizontal flip (echo views are sometimes mirrored)
            if np.random.rand() < 0.5:
                frame = np.fliplr(frame).copy()
                pts   = pts.copy()
                pts[:, 0] = orig_w - pts[:, 0]

            # Small brightness jitter
            frame = np.clip(frame + np.random.uniform(-0.1, 0.1), 0, 1)

        # ── mask ──────────────────────────────────────────────────────────
        mask = contour_to_mask(pts, orig_w, orig_h, self.img_size)

        x = torch.from_numpy(frame).unsqueeze(0)          # (1, H, W)
        y = torch.from_numpy(mask).long()                  # (H, W)

        return x, y


# ═════════════════════════════════════════════════════════════════════════════
# 3. LOSS  — Dice + CE
# ═════════════════════════════════════════════════════════════════════════════

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss over the LV class only."""

    probs = torch.softmax(logits, dim=1)[:, LV_LABEL]   # (B, H, W)
    tgt   = (targets == LV_LABEL).float()                 # (B, H, W)

    intersection = (probs * tgt).sum(dim=(1, 2))
    union        = probs.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2))

    return 1.0 - ((2.0 * intersection + smooth) / (union + smooth)).mean()


ce_loss = nn.CrossEntropyLoss(
    weight=torch.tensor([1.0] + [8.0] + [1.0] * (NUM_CLASSES - 2)).to(DEVICE)
    if DEVICE == "cpu" else None   # will move to device in training loop
)


def seg_loss(logits, targets, device):
    """Combined CE + Dice."""
    if isinstance(logits, tuple): logits = logits[0]  # unwrap deep supervision

    w = torch.tensor([1.0] + [8.0] + [1.0] * (NUM_CLASSES - 2)).to(device)
    ce = nn.functional.cross_entropy(logits, targets, weight=w)
    dc = dice_loss(logits, targets)

    return ce + dc


# ═════════════════════════════════════════════════════════════════════════════
# 4. TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def dice_score(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean Dice over the LV class (for validation metric)."""
    if isinstance(logits, tuple): logits = logits[0]

    with torch.no_grad():
        preds = torch.argmax(logits, dim=1)
        pred_lv = (preds == LV_LABEL).float()
        tgt_lv  = (targets == LV_LABEL).float()

        intersection = (pred_lv * tgt_lv).sum(dim=(1, 2))
        union        = pred_lv.sum(dim=(1, 2)) + tgt_lv.sum(dim=(1, 2))

        dice = ((2.0 * intersection + 1) / (union + 1)).mean().item()

    return dice


def train(model, loader, optimizer, scaler, device):

    model.train()
    total_loss = 0
    total_dice = 0

    for x, y in tqdm(loader, leave=False):

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(x)
            loss   = seg_loss(logits, y, device)

        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_dice += dice_score(logits, y)

    n = len(loader)
    return total_loss / n, total_dice / n


def validate(model, loader, device):

    model.eval()
    total_loss = 0
    total_dice = 0

    with torch.no_grad():
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                out    = model(x)
                if isinstance(out, tuple):          # deep supervision
                    logits, aux = out[0], out[1]
                    loss = seg_loss(logits, y, device)
                    for a in aux:
                        loss = loss + 0.4 * seg_loss(a, y, device)
                else:
                    logits = out
                    loss   = seg_loss(logits, y, device)
            total_loss += loss.item()
            total_dice += dice_score(logits, y)

    n = len(loader)
    return total_loss / n, total_dice / n


# ═════════════════════════════════════════════════════════════════════════════
# 5. VOLUME COMPUTATION  (post-training evaluation)
# ═════════════════════════════════════════════════════════════════════════════

def _smooth(areas: np.ndarray) -> np.ndarray:

    n      = len(areas)
    window = min(11, n if n % 2 == 1 else n - 1)
    window = max(window, 5)
    if window % 2 == 0:
        window -= 1
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(areas, window_length=window, polyorder=3)
    except Exception:
        k = max(3, window // 2)
        return np.convolve(areas, np.ones(k) / k, mode="same")


def _find_ed_es(smoothed: np.ndarray):

    prominence  = max(10.0, 0.05 * np.ptp(smoothed))
    ed_cands, _ = find_peaks( smoothed, prominence=prominence)
    es_cands, _ = find_peaks(-smoothed, prominence=prominence)

    best_ed   = int(np.argmax(smoothed))
    best_es   = int(np.argmin(smoothed))
    best_diff = -np.inf

    if ed_cands.size and es_cands.size:
        for ed in ed_cands:
            for es in es_cands:
                if abs(int(ed) - int(es)) < 8:
                    continue
                diff = smoothed[ed] - smoothed[es]
                if diff > best_diff:
                    best_diff = diff
                    best_ed, best_es = int(ed), int(es)

    return best_ed, best_es


def compute_volumes(model: nn.Module, video_path: str, device: str):
    """
    Run UNet on every frame, extract LV area curve, detect ED/ES,
    compute EDV/ESV/SV/EF using the A^1.5 volume proxy.

    Returns dict with EDV_mL, ESV_mL, SV_mL, EF_pct (all floats).
    """

    cap   = cv2.VideoCapture(video_path)
    areas = []

    model.eval()

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            x     = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(device)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(x)
            mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

            areas.append(float(np.sum(mask == LV_LABEL)))

    cap.release()

    if len(areas) < 5:
        return None

    areas    = np.array(areas)
    smoothed = _smooth(areas)
    ed, es   = _find_ed_es(smoothed)

    # A^1.5 volume proxy (cancels in ratio, better captures 3-D volume swing)
    ed_vol = smoothed[ed] ** 1.5
    es_vol = smoothed[es] ** 1.5

    if ed_vol <= 0:
        return None

    ef  = float(np.clip((ed_vol - es_vol) / ed_vol * 100.0, 0, 100))

    # Scale to mL using a rough calibration factor derived from EchoNet stats
    # (median EDV ~110 mL, median LV area at ED ~3500 px at 384px resolution)
    # Factor k: EDV_mL = k * area_px^1.5  →  k ≈ 110 / 3500^1.5 ≈ 5.3e-4
    K_ML = 2.96e-5
    edv  = float(ed_vol * K_ML)
    esv  = float(es_vol * K_ML)
    sv   = edv - esv

    return {
        "EDV_mL":  edv,
        "ESV_mL":  esv,
        "SV_mL":   sv,
        "EF_pct":  ef,
        "ed_frame": ed,
        "es_frame": es,
    }


def evaluate_volumes(model: nn.Module, data_root: str, device: str, n: int = 100):
    """
    Evaluate EDV/ESV/EF on the test split against FileList.csv ground truth.
    EchoNet provides EF, EDV, ESV directly in FileList.csv.
    """

    file_list = pd.read_csv(os.path.join(data_root, "FileList.csv"))
    test_df   = file_list[file_list["Split"] == "TEST"].reset_index(drop=True)

    if n is not None:
        test_df = test_df.head(n)

    video_dir = os.path.join(data_root, "Videos")

    ef_errors, edv_errors, esv_errors = [], [], []

    print(f"\nEvaluating volumes on {len(test_df)} test videos...\n")

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):

        fname      = str(row["FileName"]).replace(".avi", "")
        video_path = os.path.join(video_dir, fname + ".avi")

        if not os.path.exists(video_path):
            continue

        result = compute_volumes(model, video_path, device)

        if result is None:
            continue

        true_ef  = float(row["EF"])
        true_edv = float(row["EDV"]) if "EDV" in row else float("nan")
        true_esv = float(row["ESV"]) if "ESV" in row else float("nan")

        ef_errors.append(abs(result["EF_pct"] - true_ef))

        if not np.isnan(true_edv):
            edv_errors.append(abs(result["EDV_mL"] - true_edv))
        if not np.isnan(true_esv):
            esv_errors.append(abs(result["ESV_mL"] - true_esv))

        print(
            f"{fname}  |  "
            f"EF  true={true_ef:.1f}  pred={result['EF_pct']:.1f}  |  "
            f"EDV true={true_edv:.1f}  pred={result['EDV_mL']:.1f}  |  "
            f"ESV true={true_esv:.1f}  pred={result['ESV_mL']:.1f}  |  "
            f"SV={result['SV_mL']:.1f}"
        )

    print(f"\n{'='*60}")
    if ef_errors:
        print(f"EF  MAE: {np.mean(ef_errors):.2f}  ±{np.std(ef_errors):.2f}")
    if edv_errors:
        print(f"EDV MAE: {np.mean(edv_errors):.2f}  ±{np.std(edv_errors):.2f} mL")
    if esv_errors:
        print(f"ESV MAE: {np.mean(esv_errors):.2f}  ±{np.std(esv_errors):.2f} mL")
    print(f"{'='*60}\n")


# ═════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main(eval_only: bool = False):

    tracings = parse_volume_tracings(DATA_ROOT)

    model = UNet(in_ch=1, num_classes=NUM_CLASSES).to(DEVICE)

    if eval_only or os.path.exists(CKPT_OUT):
        state = torch.load(CKPT_OUT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state)
        print(f"Loaded checkpoint from {CKPT_OUT}")

        if eval_only:
            evaluate_volumes(model, DATA_ROOT, DEVICE)
            return

    # ── datasets ──────────────────────────────────────────────────────────
    train_ds = EchoSegDataset(DATA_ROOT, "train", tracings)
    val_ds   = EchoSegDataset(DATA_ROOT, "val",   tracings)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # ── optimiser + schedule ──────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)
    scaler    = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_dice         = 0.0
    epochs_no_improve = 0

    print(f"\nTraining for up to {MAX_EPOCHS} epochs (patience={PATIENCE})\n")

    for epoch in range(MAX_EPOCHS):

        tr_loss, tr_dice = train(model, train_loader, optimizer, scaler, DEVICE)
        vl_loss, vl_dice = validate(model, val_loader, DEVICE)

        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch:02d}]  "
            f"train loss {tr_loss:.4f}  dice {tr_dice:.4f}  |  "
            f"val loss {vl_loss:.4f}  dice {vl_dice:.4f}  |  "
            f"lr {lr:.1e}"
        )

        if vl_dice > best_dice:
            best_dice         = vl_dice
            epochs_no_improve = 0
            torch.save(model.state_dict(), CKPT_OUT)
            print(f"  ↑ Best val Dice {best_dice:.4f} — saved to {CKPT_OUT}")

        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping (no Dice improvement for {PATIENCE} epochs)")
                break

    print(f"\nTraining done. Best val Dice: {best_dice:.4f}")

    # ── post-training volume evaluation ───────────────────────────────────
    model.load_state_dict(torch.load(CKPT_OUT, map_location=DEVICE, weights_only=False))
    evaluate_volumes(model, DATA_ROOT, DEVICE, n=100)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, load checkpoint and run volume evaluation")
    args = parser.parse_args()

    main(eval_only=args.eval_only)
