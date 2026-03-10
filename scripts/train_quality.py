#!/usr/bin/env python

"""
CACTUS Quality + Grade Training (0–9 grading)

Outputs
-------
grade: integer 0–9
quality:
    0 = Bad
    1 = Okay
    2 = Good
"""

import sys
sys.modules["zstandard"] = None

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from PIL import Image
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = "/scratch/users/joshua04/ECHO-Codex"
DATA_ROOT = "/scratch/users/joshua04/ECHO/data/cactus/cactus_dataset"

GRADES_DIR = os.path.join(DATA_ROOT, "Grades")
IMAGES_DIR = os.path.join(DATA_ROOT, "Images_Dataset")

print("Dataset root:", DATA_ROOT)


# ---------------------------------------------------------
# Grade → quality bucket
# ---------------------------------------------------------

def grade_to_quality(grade):

    grade = int(grade)

    if grade <= 3:
        return 0
    elif grade <= 6:
        return 1
    else:
        return 2


# ---------------------------------------------------------
# Load CSV grading files
# ---------------------------------------------------------

csv_files = glob.glob(os.path.join(GRADES_DIR, "*.csv"))

if not csv_files:
    csv_files = glob.glob(os.path.join(GRADES_DIR, "**/*.csv"), recursive=True)

print("Found CSV files:", len(csv_files))

dfs = []

for csv in csv_files:
    df = pd.read_csv(csv)
    dfs.append(df)

grades_df = pd.concat(dfs, ignore_index=True)

print("Total rows:", len(grades_df))


# ---------------------------------------------------------
# Build dataset table
# ---------------------------------------------------------

records = []

for _, row in grades_df.iterrows():

    view = row["Subfolder Name"]
    img_name = row["Image Name"]
    grade = int(row["Grade"])

    img_path = os.path.join(IMAGES_DIR, view, img_name)

    if os.path.exists(img_path):

        records.append({
            "path": img_path,
            "grade": grade,
            "view": view
        })

data_df = pd.DataFrame(records)

print("Matched images:", len(data_df))


# ---------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------

train_df, val_df = train_test_split(
    data_df,
    test_size=0.2,
    random_state=42,
    stratify=data_df["view"]
)

print("Train:", len(train_df))
print("Val:", len(val_df))


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

class CactusDataset(Dataset):

    def __init__(self, df, transform=None):

        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.loc[idx]

        img = Image.open(row["path"]).convert("RGB")

        grade = float(row["grade"])

        quality = grade_to_quality(grade)

        if self.transform:
            img = self.transform(img)

        grade = torch.tensor(grade, dtype=torch.float32)
        quality = torch.tensor(quality, dtype=torch.long)

        return img, grade, quality


# ---------------------------------------------------------
# Augmentations
# ---------------------------------------------------------

train_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),
    transforms.RandomAffine(
        degrees=5,
        translate=(0.02,0.02)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

val_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


# ---------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------

train_ds = CactusDataset(train_df, train_tfms)
val_ds = CactusDataset(val_df, val_tfms)

train_loader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=64,
    num_workers=8,
    pin_memory=True
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

class QualityGradeModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

        feat_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.grade_head = nn.Sequential(
            nn.Linear(feat_dim,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,1)
        )

        self.quality_head = nn.Sequential(
            nn.Linear(feat_dim,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,3)
        )

    def forward(self,x):

        features = self.backbone(x)

        grade = self.grade_head(features)

        quality_logits = self.quality_head(features)

        return grade, quality_logits


model = QualityGradeModel().to(device)


# ---------------------------------------------------------
# Class weights
# ---------------------------------------------------------

quality_counts = np.bincount(
    train_df["grade"].apply(grade_to_quality)
)

weights = 1.0 / quality_counts
weights = weights / weights.sum()

weights = torch.tensor(weights, dtype=torch.float32).to(device)


# ---------------------------------------------------------
# Loss functions
# ---------------------------------------------------------

grade_loss_fn = nn.HuberLoss()

quality_loss_fn = nn.CrossEntropyLoss(weight=weights)


# ---------------------------------------------------------
# Optimizer
# ---------------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=30
)


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

os.makedirs(os.path.join(PROJECT_ROOT, "checkpoints"), exist_ok=True)

best_val = 1e9

best_model_path = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "quality_best.pth"
)


for epoch in range(30):

    print("\nEpoch", epoch)

    model.train()

    train_loss = 0

    for imgs, grades, qualities in tqdm(train_loader):

        imgs = imgs.to(device)
        grades = grades.to(device).unsqueeze(1)
        qualities = qualities.to(device)

        pred_grade, pred_quality = model(imgs)

        loss_grade = grade_loss_fn(pred_grade, grades)

        loss_quality = quality_loss_fn(pred_quality, qualities)

        loss = loss_grade + 0.7 * loss_quality

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    scheduler.step()


    model.eval()

    val_loss = 0

    grade_errors = []
    correct_quality = 0
    total_quality = 0

    with torch.no_grad():

        for imgs, grades, qualities in val_loader:

            imgs = imgs.to(device)
            grades = grades.to(device).unsqueeze(1)
            qualities = qualities.to(device)

            pred_grade, pred_quality = model(imgs)

            loss_grade = grade_loss_fn(pred_grade, grades)

            loss_quality = quality_loss_fn(pred_quality, qualities)

            loss = loss_grade + 0.7 * loss_quality

            val_loss += loss.item()

            pred_grade_round = torch.round(pred_grade)

            grade_errors.extend(
                torch.abs(pred_grade_round - grades).cpu().numpy()
            )

            preds = pred_quality.argmax(1)

            correct_quality += (preds == qualities).sum().item()

            total_quality += qualities.size(0)

    val_loss /= len(val_loader)

    mae = sum(grade_errors)/len(grade_errors)

    quality_acc = correct_quality / total_quality

    print("Train loss:", train_loss)
    print("Val loss:", val_loss)
    print("Grade MAE:", mae)
    print("Quality accuracy:", quality_acc)


    if val_loss < best_val:

        best_val = val_loss

        torch.save(model.state_dict(), best_model_path)

        print("Saved best model")


# ---------------------------------------------------------
# Save final
# ---------------------------------------------------------

final_path = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "quality_final.pth"
)

torch.save(model.state_dict(), final_path)

print("\nTraining complete")
print("Final model:", final_path)