"""
datasets/cardiac_uda.py — Improved CardiacUDA Dataset
=====================================================

Key changes from original:
  1. Per-frame z-score normalization with percentile clipping
     → SINGLE BIGGEST improvement for cross-domain generalization.
       Different ultrasound machines produce different intensity
       distributions. Z-score removes this machine-specific bias.
  2. Stronger augmentation: rotation, scaling, intensity jitter, noise
     → Reduces overfitting on small source domain (~150 volumes)
  3. Volume caching to reduce repeated NIfTI I/O
  4. Percentile clipping (1st–99th) before normalization
     → Handles outlier pixels from ultrasound gain artifacts
"""

import os
import glob
import math
import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class CardiacUDADataset(Dataset):
    def __init__(
        self,
        data_root,
        domain="G",
        resize=384,
        augment=False,
        normalize_mode="zscore",   # "zscore" | "minmax" | "none"
    ):
        """
        Parameters
        ----------
        data_root : str
            Path to cardiacUDC_dataset/.
        domain : str
            "G" for source (Site_G_*), "R" for target (Site_R_*).
        resize : int
            Output spatial size (resize × resize).
        augment : bool
            Apply training augmentation (rotation, intensity jitter, etc).
        normalize_mode : str
            "zscore" — per-frame zero-mean unit-variance (recommended).
            "minmax" — per-frame [0, 1] normalization.
            "none"   — raw intensities (NOT recommended).
        """
        self.data_root = data_root
        self.domain = domain
        self.resize = resize
        self.augment = augment
        self.normalize_mode = normalize_mode

        # ---- Discover image/label pairs ----
        site_dirs = sorted(glob.glob(
            os.path.join(data_root, f"Site_{domain}_*")
        ))
        self.samples = []
        for site in site_dirs:
            image_files = sorted(glob.glob(os.path.join(site, "*_image.nii.gz")))
            for img_path in image_files:
                label_path = img_path.replace("_image.nii.gz", "_label.nii.gz")
                if os.path.exists(label_path):
                    self.samples.append((img_path, label_path))

        if not self.samples:
            raise FileNotFoundError(
                f"No image/label pairs for domain={domain} in {data_root}. "
                f"Site dirs: {site_dirs}"
            )

        print(f"[CardiacUDA] Domain {domain}: {len(self.samples)} volumes | "
              f"augment={augment} | norm={normalize_mode}")

        # ---- Volume cache (avoids rereading NIfTI every call) ----
        self._cache = {}
        self._use_cache = len(self.samples) <= 60

    def __len__(self):
        return len(self.samples)

    # ==================================================================
    # Loading + Caching
    # ==================================================================

    def _load_volume(self, idx):
        if self._use_cache and idx in self._cache:
            return self._cache[idx]

        img_path, label_path = self.samples[idx]
        try:
            img_data = nib.load(img_path).get_fdata(dtype=np.float32)
            lbl_data = nib.load(label_path).get_fdata(dtype=np.float32)
        except (nib.filebasedimages.ImageFileError, EOFError, Exception) as e:
            print(f"WARNING: Corrupted file {img_path}: {e}")
            return None

        if self._use_cache:
            self._cache[idx] = (img_data, lbl_data)
        return (img_data, lbl_data)

    # ==================================================================
    # Per-frame normalization
    # ==================================================================

    def _normalize_frame(self, img):
        """
        Per-frame intensity normalization.

        WHY THIS MATTERS:
        Your original code sends raw NIfTI intensities to the model.
        Site_G ultrasound machines might produce values in [0, 200],
        Site_R machines might produce [0, 800].  The model memorises
        the source distribution → fails on target.

        Z-score normalization maps every frame to mean=0, std=1,
        making the model intensity-invariant.
        """
        if self.normalize_mode == "zscore":
            # Percentile clip: remove outlier pixels (probe edges, gain artifacts)
            p1, p99 = np.percentile(img, [1, 99])
            img = np.clip(img, p1, p99)
            mu = img.mean()
            sd = img.std()
            if sd > 1e-6:
                img = (img - mu) / sd
            else:
                img = img - mu
            # Soft-clip to [-3, 3] for numerical stability
            img = np.clip(img, -3.0, 3.0)

        elif self.normalize_mode == "minmax":
            p1, p99 = np.percentile(img, [1, 99])
            img = np.clip(img, p1, p99)
            lo, hi = img.min(), img.max()
            if hi - lo > 1e-6:
                img = (img - lo) / (hi - lo)
            else:
                img = np.zeros_like(img)
        # "none" → pass through

        return img

    # ==================================================================
    # Augmentation
    # ==================================================================

    def _augment_frame(self, img, mask):
        """
        Augmentation pipeline for ultrasound cardiac images.

        Included (principled reasons):
          • Horizontal flip — anatomy can appear mirrored by probe orientation
          • Small rotation ±15° — probe angle variation
          • Small scale ±10% — depth variation
          • Intensity jitter — gain/brightness variation between machines
          • Gaussian noise — approximates speckle noise

        NOT included (would hurt):
          • Large rotation — cardiac anatomy has preferred orientation
          • Elastic deformation — creates anatomically impossible shapes
          • Color jitter — images are grayscale
          • Vertical flip — heart is always at top of sector in A4C
        """
        # Horizontal flip (50%)
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        # Rotation ±15° (40%)
        if random.random() > 0.6:
            angle = random.uniform(-15, 15)
            img, mask = self._rotate(img, mask, angle)

        # Scale ±10% (30%)
        if random.random() > 0.7:
            scale = random.uniform(0.9, 1.1)
            img, mask = self._scale(img, mask, scale)

        # Intensity jitter — image only (50%)
        if random.random() > 0.5:
            img = img * random.uniform(0.85, 1.15)
            img = img + random.uniform(-0.1, 0.1)

        # Gaussian noise — image only (30%)
        if random.random() > 0.7:
            sigma = random.uniform(0.02, 0.08)
            img = img + np.random.normal(0, sigma, img.shape).astype(np.float32)

        return img, mask

    def _rotate(self, img, mask, angle_deg):
        """Affine rotation using torch.nn.functional (no OpenCV dependency)."""
        h, w = img.shape
        img_t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        theta = torch.tensor([[cos_a, -sin_a, 0],
                               [sin_a,  cos_a, 0]], dtype=torch.float32).unsqueeze(0)

        grid = F.affine_grid(theta, img_t.shape, align_corners=False)
        img_r = F.grid_sample(img_t, grid, mode="bilinear",
                               padding_mode="zeros", align_corners=False)
        mask_r = F.grid_sample(mask_t, grid, mode="nearest",
                                padding_mode="zeros", align_corners=False)
        return img_r.squeeze().numpy(), mask_r.squeeze().numpy().astype(np.int64)

    def _scale(self, img, mask, factor):
        """Affine scaling using torch.nn.functional."""
        img_t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

        theta = torch.tensor([[1.0 / factor, 0, 0],
                               [0, 1.0 / factor, 0]], dtype=torch.float32).unsqueeze(0)

        grid = F.affine_grid(theta, img_t.shape, align_corners=False)
        img_s = F.grid_sample(img_t, grid, mode="bilinear",
                               padding_mode="zeros", align_corners=False)
        mask_s = F.grid_sample(mask_t, grid, mode="nearest",
                                padding_mode="zeros", align_corners=False)
        return img_s.squeeze().numpy(), mask_s.squeeze().numpy().astype(np.int64)

    # ==================================================================
    # __getitem__
    # ==================================================================

    def __getitem__(self, idx):
        result = self._load_volume(idx)
        if result is None:
            return self.__getitem__(random.randint(0, len(self.samples) - 1))

        img_vol, lbl_vol = result

        # ---- Smart frame selection: prefer annotated frames ----
        T = img_vol.shape[2]
        valid_frames = [t for t in range(T) if lbl_vol[:, :, t].sum() > 0]
        frame_idx = random.choice(valid_frames) if valid_frames else random.randint(0, T - 1)

        img = img_vol[:, :, frame_idx].copy()
        mask = lbl_vol[:, :, frame_idx].copy()

        # ---- Normalize BEFORE augmentation ----
        img = self._normalize_frame(img)
        img = img.astype(np.float32)
        mask = mask.astype(np.int64)

        # ---- Augmentation ----
        if self.augment:
            img, mask = self._augment_frame(img, mask)

        # ---- To tensor + resize ----
        img_t = torch.from_numpy(img.copy()).float().unsqueeze(0)   # (1, H, W)
        mask_t = torch.from_numpy(mask.copy()).long()                # (H, W)

        img_t = F.interpolate(
            img_t.unsqueeze(0), size=(self.resize, self.resize),
            mode="bilinear", align_corners=False
        ).squeeze(0)

        mask_t = F.interpolate(
            mask_t.unsqueeze(0).unsqueeze(0).float(),
            size=(self.resize, self.resize), mode="nearest"
        ).squeeze(0).squeeze(0).long()

        mask_t = mask_t.clamp(0, 4)

        return img_t, mask_t
