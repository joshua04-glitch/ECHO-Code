#!/usr/bin/env python
"""
cardiac_report.py
─────────────────
Produces a full hemodynamic report for an EchoNet-Dynamic video (or a
whole split) by combining:
  1. EF regression model  (ef_best.pth)       → EF
  2. LV segmentation model (echo_seg_best.pth) → EDV, ESV, SV, HR

Derived metrics (no extra model needed):
  CO  = SV × HR / 1000          (L/min)
  CI  = CO / BSA                 (L/min/m²)   requires --weight-kg --height-cm
  SVI = SV / BSA                 (mL/m²)
  EDVI = EDV / BSA               (mL/m²)
  ESVI = ESV / BSA               (mL/m²)

Usage:
    # single video
    python scripts/cardiac_report.py --video path/to/video.avi

    # whole test split
    python scripts/cardiac_report.py --split TEST

    # with patient demographics for indexed metrics
    python scripts/cardiac_report.py --split TEST --weight-kg 70 --height-cm 170

    # with known MAP and CVP for SVR
    python scripts/cardiac_report.py --video foo.avi --map-mmhg 93 --cvp-mmhg 8
"""

import sys
sys.modules["zstandard"] = None

import os
import cv2
import math
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scipy.signal import savgol_filter, find_peaks
from torchvision.transforms import v2
from torchvision import tv_tensors
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────

PROJECT_ROOT = os.environ.get("ECHOML_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DATA_ROOT    = os.environ.get("ECHONET_ROOT", "/scratch/users/joshua04/ECHO/data/echonet_dynamic")
WEIGHTS_DIR  = os.path.join(PROJECT_ROOT, "weights")
REG_CKPT     = os.path.join(PROJECT_ROOT, "checkpoints", "ef_best.pth")
SEG_CKPT     = os.path.join(PROJECT_ROOT, "checkpoints", "echo_seg_best.pth")

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from src.models.ef.ef_model import EFModel
from src.models.unet import UNet

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_LEN    = 32
IMG_SIZE    = 224
SEG_SIZE    = 384
LV_LABEL    = 1
NUM_CLASSES = 5
K_ML        = 2.96e-5      # calibrated area→volume constant

# ═════════════════════════════════════════════════════════════════════════════
# Model loading
# ═════════════════════════════════════════════════════════════════════════════

def load_reg_model():
    model = EFModel(
        arch="convnext_tiny", n_heads=8, n_layers=4,
        clip_len=CLIP_LEN, pretrained=True, weights_dir=WEIGHTS_DIR,
    ).to(DEVICE)
    state = torch.load(REG_CKPT, map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    print(f"  Regression  : {REG_CKPT}")
    return model


def load_seg_model():
    model = UNet(in_ch=1, num_classes=NUM_CLASSES).to(DEVICE)
    state = torch.load(SEG_CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    print(f"  Segmentation: {SEG_CKPT}")
    return model


# ═════════════════════════════════════════════════════════════════════════════
# EF regression inference (multi-clip TTA)
# ═════════════════════════════════════════════════════════════════════════════

_reg_transform = v2.Compose([
    v2.CenterCrop(size=(IMG_SIZE, IMG_SIZE)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_clips(video_path, n_clips=4):
    cap         = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    max_start = max(frame_count - CLIP_LEN, 1)
    starts    = np.linspace(0, max_start, n_clips, dtype=int)
    clips     = []

    for start in starts:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(start) - 1, 0))
        frames = []
        for _ in range(CLIP_LEN):
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (256, 256))
                frames.append(frame)
            else:
                frames.append(frames[-1])
        cap.release()
        v = np.stack(frames)
        v = tv_tensors.Video(np.transpose(v, (0, 3, 1, 2)))
        v = _reg_transform(v)
        v = v.permute(1, 0, 2, 3)   # (C, T, H, W)
        clips.append(v)

    return torch.stack(clips, dim=0), fps   # (n_clips, C, T, H, W)


def predict_ef_regression(reg_model, video_path):
    clips, fps = _load_clips(video_path)
    clips = clips.to(DEVICE)
    use_amp = DEVICE == "cuda"
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=use_amp):
            preds = reg_model(clips)
    return float(preds.mean().cpu())


# ═════════════════════════════════════════════════════════════════════════════
# Segmentation inference → volumes + HR
# ═════════════════════════════════════════════════════════════════════════════

def _smooth(areas):
    n      = len(areas)
    window = min(11, n if n % 2 == 1 else n - 1)
    window = max(window, 5)
    if window % 2 == 0:
        window -= 1
    try:
        return savgol_filter(areas, window_length=window, polyorder=3)
    except Exception:
        k = max(3, window // 2)
        return np.convolve(areas, np.ones(k) / k, mode="same")


def _find_ed_es(smoothed):
    prominence  = max(10.0, 0.05 * np.ptp(smoothed))
    ed_cands, _ = find_peaks( smoothed, prominence=prominence)
    es_cands, _ = find_peaks(-smoothed, prominence=prominence)
    best_ed, best_es = int(np.argmax(smoothed)), int(np.argmin(smoothed))
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


def predict_volumes_and_hr(seg_model, video_path):
    """
    Returns dict with EDV, ESV, SV (mL) and HR (bpm) estimated from video.
    HR is derived from the spacing between successive ED peaks in the area curve.
    """

    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    areas = []

    use_amp = DEVICE == "cuda"

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (SEG_SIZE, SEG_SIZE))
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            x     = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = seg_model(x)
            logits = out[0] if isinstance(out, tuple) else out
            mask   = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
            areas.append(float(np.sum(mask == LV_LABEL)))
    cap.release()

    if len(areas) < 5:
        return None

    areas    = np.array(areas)
    smoothed = _smooth(areas)
    ed, es   = _find_ed_es(smoothed)

    ed_vol = smoothed[ed] ** 1.5
    es_vol = smoothed[es] ** 1.5

    if ed_vol <= 0:
        return None

    edv = float(ed_vol * K_ML)
    esv = float(es_vol * K_ML)
    sv  = edv - esv
    ef  = float(np.clip((ed_vol - es_vol) / ed_vol * 100.0, 0, 100))

    # ── Heart Rate from peak spacing ──────────────────────────────────────
    # Find all ED peaks, compute mean inter-peak interval in seconds → bpm
    prominence  = max(10.0, 0.05 * np.ptp(smoothed))
    ed_peaks, _ = find_peaks(smoothed, prominence=prominence, distance=int(fps * 0.3))

    if len(ed_peaks) >= 2:
        mean_interval_frames = np.diff(ed_peaks).mean()
        hr = float(60.0 * fps / mean_interval_frames)
        hr = float(np.clip(hr, 30, 200))   # physiological sanity check
    else:
        # Fallback: assume typical adult resting HR
        hr = 70.0

    return {
        "EDV_mL":   edv,
        "ESV_mL":   esv,
        "SV_mL":    sv,
        "EF_seg":   ef,
        "HR_bpm":   hr,
        "ed_frame": ed,
        "es_frame": es,
        "fps":      fps,
    }


# ═════════════════════════════════════════════════════════════════════════════
# BSA and derived metrics
# ═════════════════════════════════════════════════════════════════════════════

def bsa_dubois(weight_kg, height_cm):
    """DuBois & DuBois formula (most widely used in cardiology)."""
    return 0.007184 * (weight_kg ** 0.425) * (height_cm ** 0.725)


def compute_hemodynamics(vols, ef_reg, weight_kg=None, height_cm=None,
                         map_mmhg=None, cvp_mmhg=8):
    """
    Combine regression EF + segmentation volumes into a full report dict.
    Optional inputs: patient demographics (for indexed metrics),
                     MAP + CVP (for SVR).
    """

    sv  = vols["SV_mL"]
    hr  = vols["HR_bpm"]
    co  = sv * hr / 1000.0   # L/min

    result = {
        # ── Volumes ───────────────────────────────────────────────────────
        "EDV_mL":          round(vols["EDV_mL"], 1),
        "ESV_mL":          round(vols["ESV_mL"], 1),
        "SV_mL":           round(sv, 1),

        # ── EF (two estimates) ────────────────────────────────────────────
        "EF_regression_%": round(ef_reg, 1),
        "EF_seg_%":        round(vols["EF_seg"], 1),

        # ── Flow ──────────────────────────────────────────────────────────
        "HR_bpm":          round(hr, 1),
        "CO_L_min":        round(co, 2),

        # ── Timing ────────────────────────────────────────────────────────
        "ED_frame":        vols["ed_frame"],
        "ES_frame":        vols["es_frame"],
    }

    # ── Indexed metrics (require demographics) ────────────────────────────
    if weight_kg is not None and height_cm is not None:
        bsa = bsa_dubois(weight_kg, height_cm)
        result["BSA_m2"]       = round(bsa, 3)
        result["CI_L_min_m2"]  = round(co / bsa, 2)
        result["SVI_mL_m2"]    = round(sv / bsa, 1)
        result["EDVI_mL_m2"]   = round(vols["EDV_mL"] / bsa, 1)
        result["ESVI_mL_m2"]   = round(vols["ESV_mL"] / bsa, 1)

    # ── Vascular resistance (requires MAP + CVP) ──────────────────────────
    if map_mmhg is not None and co > 0:
        svr = ((map_mmhg - cvp_mmhg) / co) * 80   # dynes·s/cm⁵
        result["MAP_mmHg"]             = map_mmhg
        result["CVP_mmHg"]             = cvp_mmhg
        result["SVR_dynes_s_cm5"]      = round(svr, 0)

    return result


# ═════════════════════════════════════════════════════════════════════════════
# Report printer
# ═════════════════════════════════════════════════════════════════════════════

def print_report(video_name, report, true_ef=None):

    W = 56
    print("\n" + "═" * W)
    print(f"  CARDIAC REPORT — {os.path.basename(video_name)}")
    print("═" * W)

    print("\n  ── Ejection Fraction ──────────────────────────────")
    if true_ef is not None:
        print(f"  True EF          : {true_ef:.1f} %")
    print(f"  EF (regression)  : {report['EF_regression_%']} %  ← primary estimate")
    print(f"  EF (seg-based)   : {report['EF_seg_%']} %")

    print("\n  ── Volumes ────────────────────────────────────────")
    print(f"  EDV              : {report['EDV_mL']} mL")
    print(f"  ESV              : {report['ESV_mL']} mL")
    print(f"  Stroke Volume    : {report['SV_mL']} mL")
    print(f"  ED frame         : {report['ED_frame']}")
    print(f"  ES frame         : {report['ES_frame']}")

    print("\n  ── Cardiac Output ─────────────────────────────────")
    print(f"  Heart Rate       : {report['HR_bpm']} bpm")
    print(f"  Cardiac Output   : {report['CO_L_min']} L/min")

    if "BSA_m2" in report:
        print("\n  ── Indexed Metrics ────────────────────────────────")
        print(f"  BSA              : {report['BSA_m2']} m²")
        print(f"  Cardiac Index    : {report['CI_L_min_m2']} L/min/m²")
        print(f"  SV Index         : {report['SVI_mL_m2']} mL/m²")
        print(f"  EDV Index        : {report['EDVI_mL_m2']} mL/m²")
        print(f"  ESV Index        : {report['ESVI_mL_m2']} mL/m²")

    if "SVR_dynes_s_cm5" in report:
        print("\n  ── Haemodynamics ──────────────────────────────────")
        print(f"  MAP              : {report['MAP_mmHg']} mmHg")
        print(f"  CVP              : {report['CVP_mmHg']} mmHg")
        print(f"  SVR              : {report['SVR_dynes_s_cm5']} dynes·s/cm⁵")

    print("═" * W + "\n")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def run(args):

    print("\nLoading models...")
    reg_model = load_reg_model()
    seg_model = load_seg_model()

    # ── Collect videos ────────────────────────────────────────────────────
    if args.video:
        videos    = [{"path": args.video, "name": args.video, "true_ef": None}]
    else:
        df = pd.read_csv(os.path.join(DATA_ROOT, "FileList.csv"))
        df = df[df["Split"] == args.split.upper()].reset_index(drop=True)
        if args.n:
            df = df.head(args.n)
        video_dir = os.path.join(DATA_ROOT, "Videos")
        videos = [
            {
                "path":    os.path.join(video_dir, row["FileName"] + ".avi"),
                "name":    row["FileName"] + ".avi",
                "true_ef": float(row["EF"]),
            }
            for _, row in df.iterrows()
        ]

    ef_errors = []

    for v in tqdm(videos, desc="Processing"):

        if not os.path.exists(v["path"]):
            print(f"  [SKIP] {v['name']} not found")
            continue

        ef_reg = predict_ef_regression(reg_model, v["path"])
        vols   = predict_volumes_and_hr(seg_model, v["path"])

        if vols is None:
            print(f"  [SKIP] {v['name']} — segmentation failed")
            continue

        report = compute_hemodynamics(
            vols, ef_reg,
            weight_kg  = args.weight_kg,
            height_cm  = args.height_cm,
            map_mmhg   = args.map_mmhg,
            cvp_mmhg   = args.cvp_mmhg,
        )

        print_report(v["name"], report, true_ef=v["true_ef"])

        if v["true_ef"] is not None:
            ef_errors.append(abs(report["EF_regression_%"] - v["true_ef"]))

    if ef_errors:
        print(f"Summary — EF regression MAE: {np.mean(ef_errors):.2f} ±{np.std(ef_errors):.2f} "
              f"over {len(ef_errors)} videos\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # ── Input ──────────────────────────────────────────────────────────────
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video",    type=str, help="Path to a single .avi file")
    group.add_argument("--split",    type=str, choices=["TRAIN","VAL","TEST"],
                       help="Run on a full dataset split")

    parser.add_argument("--n",        type=int, default=None,
                        help="Limit to first N videos (split mode only)")

    # ── Patient demographics (optional, enables indexed metrics) ───────────
    parser.add_argument("--weight-kg",  type=float, default=None)
    parser.add_argument("--height-cm",  type=float, default=None)

    # ── Haemodynamic inputs (optional, enables SVR) ────────────────────────
    parser.add_argument("--map-mmhg",   type=float, default=None,
                        help="Mean arterial pressure in mmHg")
    parser.add_argument("--cvp-mmhg",   type=float, default=8.0,
                        help="Central venous pressure in mmHg (default 8)")

    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    run(args)
