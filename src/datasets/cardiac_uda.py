"""
datasets/cardiac_uda.py — Multi-source cardiac segmentation datasets.

Unified label scheme:
  0 = Background
  1 = LV  (left ventricle)   — Red
  2 = LA  (left atrium)      — Yellow
  3 = RA  (right atrium)     — Blue
  4 = RV  (right ventricle)  — Green

Source datasets and their mappings:
  CardiacUDA (G/R): 0=BG, 1=LV, 2=LA, 3=RA, 4=RV  (native, no remap needed)
  CAMUS:            0=BG, 1=LV, 3=LA                 (no RV, no RA)
  PLOSONE:          JSON contours for LV, RV, LA, RA
"""

import os
import glob
import json
import math
import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image

IGNORE_INDEX = 255


# ==============================================================
# CardiacUDA Dataset (NIfTI volumes — G, R, CAMUS domains)
# ==============================================================

class CardiacUDADataset(Dataset):

    def __init__(
        self,
        data_root,
        domain="G",
        resize=384,
        augment=False,
        normalize_mode="zscore",
    ):
        self.data_root = data_root
        self.domain = domain
        self.resize = resize
        self.augment = augment
        self.normalize_mode = normalize_mode

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
                f"No image/label pairs for domain={domain} in {data_root}"
            )

        self.is_partial_label = (domain == "CAMUS")

        print(
            f"[CardiacUDA] Domain {domain}: {len(self.samples)} volumes | "
            f"augment={augment} | norm={normalize_mode} | "
            f"partial_labels={self.is_partial_label}"
        )

        self._cache = {}
        self._use_cache = len(self.samples) <= 60

    def __len__(self):
        return len(self.samples)

    def _load_volume(self, idx):
        if self._use_cache and idx in self._cache:
            return self._cache[idx]

        img_path, label_path = self.samples[idx]
        img_data = nib.load(img_path).get_fdata(dtype=np.float32)
        lbl_data = nib.load(label_path).get_fdata(dtype=np.float32).astype(np.int64)

        if self._use_cache:
            self._cache[idx] = (img_data, lbl_data)

        return (img_data, lbl_data)

    def _remap_camus_labels(self, mask):
        """
        CAMUS raw: 0=BG, 1=LV, 2=MYO, 3=LA
        Unified:   0=BG, 1=LV, 2=LA
        MYO is ignored. No RV or RA.
        """
        out = np.full_like(mask, IGNORE_INDEX)
        out[mask == 0] = 0   # background
        out[mask == 1] = 1   # LV -> LV
        out[mask == 3] = 2   # LA -> LA (unified label 2)
        # mask==2 (MYO) stays IGNORE_INDEX
        return out

    def _normalize_frame(self, img):
        p1, p99 = np.percentile(img, [1, 99])
        img = np.clip(img, p1, p99)
        mu = img.mean()
        sd = img.std()
        img = (img - mu) / (sd + 1e-6)
        img = np.clip(img, -3.0, 3.0)
        return img

    def _augment_frame(self, img, mask):
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        if random.random() > 0.6:
            angle = random.uniform(-15, 15)
            img, mask = self._rotate(img, mask, angle)

        if random.random() > 0.7:
            scale = random.uniform(0.9, 1.1)
            img, mask = self._scale(img, mask, scale)

        if random.random() > 0.5:
            img = img * random.uniform(0.85, 1.15)
            img = img + random.uniform(-0.1, 0.1)

        if random.random() > 0.7:
            sigma = random.uniform(0.02, 0.08)
            img = img + np.random.normal(0, sigma, img.shape).astype(np.float32)

        return img, mask

    def _rotate(self, img, mask, angle_deg):
        img_t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        mask_f = mask.astype(np.float32)
        mask_t = torch.from_numpy(mask_f).unsqueeze(0).unsqueeze(0)

        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        theta = torch.tensor(
            [[cos_a, -sin_a, 0],
             [sin_a,  cos_a, 0]],
            dtype=torch.float32
        ).unsqueeze(0)

        grid = F.affine_grid(theta, img_t.shape, align_corners=False)

        img_r = F.grid_sample(img_t, grid, mode="bilinear",
                              padding_mode="zeros", align_corners=False)
        mask_r = F.grid_sample(mask_t, grid, mode="nearest",
                               padding_mode="zeros", align_corners=False)

        return img_r.squeeze().numpy(), mask_r.squeeze().numpy().astype(np.int64)

    def _scale(self, img, mask, factor):
        img_t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
        mask_f = mask.astype(np.float32)
        mask_t = torch.from_numpy(mask_f).unsqueeze(0).unsqueeze(0)

        theta = torch.tensor(
            [[1.0 / factor, 0, 0],
             [0, 1.0 / factor, 0]],
            dtype=torch.float32
        ).unsqueeze(0)

        grid = F.affine_grid(theta, img_t.shape, align_corners=False)

        img_s = F.grid_sample(img_t, grid, mode="bilinear",
                              padding_mode="zeros", align_corners=False)
        mask_s = F.grid_sample(mask_t, grid, mode="nearest",
                               padding_mode="zeros", align_corners=False)

        return img_s.squeeze().numpy(), mask_s.squeeze().numpy().astype(np.int64)

    def __getitem__(self, idx):
        img_vol, lbl_vol = self._load_volume(idx)

        T = img_vol.shape[2]
        valid_frames = [t for t in range(T) if lbl_vol[:, :, t].sum() > 0]

        if valid_frames:
            frame_idx = random.choice(valid_frames)
        else:
            frame_idx = random.randint(0, T - 1)

        img = img_vol[:, :, frame_idx].copy()
        mask = lbl_vol[:, :, frame_idx].copy().astype(np.int64)

        # G/R: labels are already 0=BG,1=LV,2=LA,3=RA,4=RV — no remap
        # CAMUS: needs remap
        if self.is_partial_label:
            mask = self._remap_camus_labels(mask)

        img = self._normalize_frame(img)

        if self.augment:
            img, mask = self._augment_frame(img, mask)

        img_t = torch.from_numpy(img).float().unsqueeze(0)
        mask_t = torch.from_numpy(mask).long()

        img_t = F.interpolate(
            img_t.unsqueeze(0),
            size=(self.resize, self.resize),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        mask_t = F.interpolate(
            mask_t.unsqueeze(0).unsqueeze(0).float(),
            size=(self.resize, self.resize),
            mode="nearest"
        ).squeeze(0).squeeze(0).long()

        valid = (mask_t >= 0) & (mask_t <= 4)
        ignored = (mask_t == IGNORE_INDEX)
        bad = ~valid & ~ignored
        mask_t[bad] = 0

        return img_t, mask_t


# ==============================================================
# PLOSONE Dataset (JPG images + JSON contour masks)
# ==============================================================

class PLOSONEDataset(Dataset):
    """
    PLOSONE cardiac echo dataset (100 images).

    Structure:
      Images/               — frame_1.jpg ... frame_100.jpg
      MasksJsonContours/    — mascara1.json ... mascara100.json

    JSON format: {"LV": [[row,col],...], "RV": [[row,col],...], ...}

    Mapped to unified scheme:
      LV -> 1, LA -> 2, RA -> 3, RV -> 4
    """

    CHAMBER_MAP = {
        "LV": 1,
        "LA": 2,
        "RA": 3,
        "RV": 4,
    }

    def __init__(self, data_root, resize=384, augment=False):
        self.data_root = data_root
        self.resize = resize
        self.augment = augment

        images_dir = os.path.join(data_root, "Images")
        masks_dir = os.path.join(data_root, "MasksJsonContours")

        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"No Images folder: {images_dir}")
        if not os.path.isdir(masks_dir):
            raise FileNotFoundError(f"No MasksJsonContours folder: {masks_dir}")

        self.samples = []

        for img_file in sorted(os.listdir(images_dir)):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            base = os.path.splitext(img_file)[0]
            parts = base.split("_")
            if len(parts) >= 2 and parts[-1].isdigit():
                num = parts[-1]
            else:
                continue

            json_path = os.path.join(masks_dir, f"mascara{num}.json")

            if os.path.exists(json_path):
                self.samples.append((
                    os.path.join(images_dir, img_file),
                    json_path
                ))

        print(
            f"[PLOSONE] {len(self.samples)} image/mask pairs | "
            f"augment={augment}"
        )

        if len(self.samples) == 0:
            img_files = [f for f in os.listdir(images_dir) if not f.startswith("desktop")][:5]
            mask_files = [f for f in os.listdir(masks_dir) if not f.startswith("desktop")][:5]
            print(f"  Sample images: {img_files}")
            print(f"  Sample masks:  {mask_files}")
            raise FileNotFoundError(
                f"No matched image/mask pairs in {data_root}"
            )

    def __len__(self):
        return len(self.samples)

    def _contours_to_mask(self, json_path, h, w):
        import cv2

        with open(json_path, "r") as f:
            data = json.load(f)

        mask = np.zeros((h, w), dtype=np.int32)

        for chamber_name, label_val in self.CHAMBER_MAP.items():
            if chamber_name not in data:
                continue

            points = data[chamber_name]

            if not isinstance(points, list) or len(points) < 3:
                continue

            pts = np.array(points, dtype=np.int32)

            if pts.ndim != 2 or pts.shape[1] != 2:
                continue

            pts_xy = np.ascontiguousarray(pts[:, ::-1])

            pts_xy[:, 0] = np.clip(pts_xy[:, 0], 0, w - 1)
            pts_xy[:, 1] = np.clip(pts_xy[:, 1], 0, h - 1)

            cv2.fillPoly(mask, [pts_xy], int(label_val))

        return mask.astype(np.int64)

    def _normalize_frame(self, img):
        p1, p99 = np.percentile(img, [1, 99])
        img = np.clip(img, p1, p99)
        mu = img.mean()
        sd = img.std()
        img = (img - mu) / (sd + 1e-6)
        img = np.clip(img, -3.0, 3.0)
        return img

    def _augment_frame(self, img, mask):
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()

        if random.random() > 0.6:
            angle = random.uniform(-15, 15)
            img_t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
            mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            theta = torch.tensor(
                [[cos_a, -sin_a, 0], [sin_a, cos_a, 0]],
                dtype=torch.float32
            ).unsqueeze(0)
            grid = F.affine_grid(theta, img_t.shape, align_corners=False)

            img = F.grid_sample(img_t, grid, mode="bilinear",
                                padding_mode="zeros", align_corners=False).squeeze().numpy()
            mask = F.grid_sample(mask_t, grid, mode="nearest",
                                 padding_mode="zeros", align_corners=False).squeeze().numpy().astype(np.int64)

        if random.random() > 0.5:
            img = img * random.uniform(0.85, 1.15)
            img = img + random.uniform(-0.1, 0.1)

        if random.random() > 0.7:
            sigma = random.uniform(0.02, 0.08)
            img = img + np.random.normal(0, sigma, img.shape).astype(np.float32)

        return img, mask

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]

        pil_img = Image.open(img_path).convert("L")
        img = np.array(pil_img, dtype=np.float32)
        h, w = img.shape

        mask = self._contours_to_mask(json_path, h, w)

        img = self._normalize_frame(img)

        if self.augment:
            img, mask = self._augment_frame(img, mask)

        img_t = torch.from_numpy(img).float().unsqueeze(0)
        mask_t = torch.from_numpy(mask).long()

        img_t = F.interpolate(
            img_t.unsqueeze(0),
            size=(self.resize, self.resize),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        mask_t = F.interpolate(
            mask_t.unsqueeze(0).unsqueeze(0).float(),
            size=(self.resize, self.resize),
            mode="nearest"
        ).squeeze(0).squeeze(0).long()

        valid = (mask_t >= 0) & (mask_t <= 4)
        ignored = (mask_t == IGNORE_INDEX)
        bad = ~valid & ~ignored
        mask_t[bad] = 0

        return img_t, mask_t