import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class QualityGradeModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Pretrained ResNet backbone
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Regression head for numeric grade
        self.grade_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Classification head for quality bucket
        self.quality_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):

        features = self.backbone(x)

        grade = self.grade_head(features)
        quality_logits = self.quality_head(features)

        return grade, quality_logits