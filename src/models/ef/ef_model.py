import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):

        super().__init__()

        pe = torch.zeros(1, max_len, d_model)

        position = torch.arange(max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):

        return x + self.pe[:, : x.size(1)]


class ImageEncoder(nn.Module):

    def __init__(self, arch="convnext_tiny"):

        super().__init__()

        if arch == "convnext_tiny":

            self.model = torchvision.models.convnext_tiny(
                weights="IMAGENET1K_V1"
            )

            self.n_features = self.model.classifier[-1].in_features

            self.model.classifier[-1] = nn.Identity()

        elif arch == "resnet18":

            self.model = torchvision.models.resnet18(
                weights="IMAGENET1K_V1"
            )

            self.n_features = self.model.fc.in_features

            self.model.fc = nn.Identity()

        else:

            raise ValueError("Unsupported architecture")

    def forward(self, x):

        return self.model(x)


class FrameTransformer(nn.Module):

    def __init__(
        self,
        arch,
        n_heads,
        n_layers,
        clip_len=32,
    ):

        super().__init__()

        self.encoder = ImageEncoder(arch)

        dim = self.encoder.n_features

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim,   # MUST match pretrained backbone
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.time_encoder = PositionalEncoding(
            d_model=dim,
            max_len=clip_len
        )

    def forward(self, x):

        b, c, t, h, w = x.shape

        x = x.reshape(b * t, c, h, w)

        feats = self.encoder(x)

        feats = feats.reshape(b, t, self.encoder.n_features)

        feats = self.time_encoder(feats)

        feats = self.transformer(feats)

        pooled = feats.mean(1)

        return pooled


class Task:

    def __init__(self, task_name, task_type, class_names, mean=np.nan):

        self.task_name = task_name
        self.task_type = task_type
        self.class_names = class_names
        self.class_indices = np.arange(class_names.size)
        self.mean = mean


class MultiTaskModel(nn.Module):

    def __init__(self, encoder, encoder_dim, tasks, fc_dropout=0):

        super().__init__()

        self.encoder = encoder

        self.tasks = tasks

        for task in tasks:

            if task.task_type == "multi-class_classification":

                head = nn.Sequential(
                    nn.Dropout(p=fc_dropout),
                    nn.Linear(encoder_dim, task.class_names.size),
                )

            else:

                head = nn.Sequential(
                    nn.Dropout(p=fc_dropout),
                    nn.Linear(encoder_dim, 1),
                )

                head[-1].bias.data[0] = task.mean

            self.add_module(task.task_name + "_head", head)

    def forward(self, x):

        x = self.encoder(x)

        out = {}

        for task in self.tasks:

            out[task.task_name] = self.get_submodule(
                task.task_name + "_head"
            )(x)

        return out


class EFHead(nn.Module):
    """
    Regression head with a residual shortcut from the bottleneck to the output.
    GELU activations are smoother than ReLU for regression tasks.
    """

    def __init__(self, in_dim: int, dropout: float = 0.3, init_bias: float = 55.6):

        super().__init__()

        # Main branch: in_dim → 256 → 64 → 1
        self.fc1    = nn.Linear(in_dim, 256)
        self.norm1  = nn.LayerNorm(256)
        self.act1   = nn.GELU()
        self.drop1  = nn.Dropout(dropout)

        self.fc2    = nn.Linear(256, 64)
        self.norm2  = nn.LayerNorm(64)
        self.act2   = nn.GELU()
        self.drop2  = nn.Dropout(dropout / 2)

        self.fc_out = nn.Linear(64, 1)
        self.fc_out.bias.data[0] = init_bias

        # Shortcut: project in_dim → 1 directly and add to output
        self.shortcut = nn.Linear(in_dim, 1, bias=False)

    def forward(self, x):

        h = self.drop1(self.act1(self.norm1(self.fc1(x))))
        h = self.drop2(self.act2(self.norm2(self.fc2(h))))

        return self.fc_out(h) + self.shortcut(x)


class EFModel(nn.Module):

    ECHONET_EF_MEAN = 55.6

    def __init__(
        self,
        arch="convnext_tiny",
        n_heads=8,
        n_layers=4,
        clip_len=32,
        fc_dropout=0.3,
        pretrained=True,
        weights_dir="",
    ):

        super().__init__()

        encoder = FrameTransformer(
            arch,
            n_heads,
            n_layers,
            clip_len,
        )

        encoder_dim = encoder.encoder.n_features

        if pretrained and weights_dir:

            import os

            tasks_path = os.path.join(weights_dir, "task_defs.pkl")

            weights_path = os.path.join(weights_dir, "pretrained_backbone.pt")

            task_dict = pd.read_pickle(tasks_path)

            task_list = [
                Task(
                    t,
                    task_dict[t]["task_type"],
                    task_dict[t]["class_names"],
                    task_dict[t]["mean"],
                )
                for t in task_dict.keys()
            ]

            full_model = MultiTaskModel(
                encoder,
                encoder_dim,
                task_list,
                fc_dropout
            )

            state = torch.load(
                weights_path,
                map_location="cpu"
            )["weights"]

            state.pop("encoder.time_encoder.pe", None)

            msg = full_model.load_state_dict(
                state,
                strict=False
            )

            print("Loaded pretrained backbone:", msg)

            self.backbone = full_model.encoder

        else:

            self.backbone = encoder

        self.head = EFHead(
            in_dim=encoder_dim,
            dropout=fc_dropout,
            init_bias=self.ECHONET_EF_MEAN,
        )

    def forward(self, x):

        feats = self.backbone(x)

        return self.head(feats)