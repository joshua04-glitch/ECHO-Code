"""
models/unet.py — Improved U-Net for Cardiac Segmentation
=========================================================

Changes from your original UNet and WHY:

1. 4 encoder levels instead of 3
   WHY: Your 384×384 input with 3 levels gives a 48×48 bottleneck.
   That's too large — the receptive field doesn't cover the full
   cardiac sector. 4 levels → 24×24 bottleneck → each bottleneck
   neuron "sees" the whole image.

2. Bilinear upsample + Conv replaces ConvTranspose2d
   WHY: ConvTranspose2d produces checkerboard artifacts — this is
   the direct cause of your boundary bleeding. Bilinear upsampling
   is smooth by construction.

3. Attention gates in decoder (Oktay et al., 2018)
   WHY: Standard skip connections pass ALL encoder features to
   decoder, including irrelevant noise regions. Attention gates
   learn to suppress activations far from cardiac structures.
   → Reduces oversegmentation in inferior wall.

4. Residual connections within conv blocks
   WHY: Better gradient flow → faster convergence → less instability
   in later epochs. The block learns a refinement (residual)
   rather than a full transformation.

5. Deep supervision (auxiliary loss heads at decoder levels 2, 3)
   WHY: Provides direct gradient signal to mid-level decoder layers.
   Without it, gradients from the final output must propagate through
   4 decoder levels — they attenuate. Deep supervision keeps ALL
   decoder levels well-trained.

6. InstanceNorm2d retained (with affine=True)
   WHY: InstanceNorm normalizes per-sample, removing machine-specific
   intensity statistics. This is the right choice for domain
   generalization. BatchNorm would memorize source domain statistics.

7. Dropout in bottleneck
   WHY: The bottleneck has the most parameters and is most prone to
   overfitting on a small source domain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResDoubleConv(nn.Module):
    """Two 3×3 convs + InstanceNorm + ReLU, with a residual skip."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
        )
        self.skip = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class AttentionGate(nn.Module):
    """
    Attention gate (Oktay et al., 2018).

    gate : from decoder (coarser, semantically rich)
    skip : from encoder (finer, spatially precise)

    Learns a (0,1) attention map that suppresses irrelevant
    skip-connection features before concatenation.
    """

    def __init__(self, gate_ch, skip_ch, inter_ch=None):
        super().__init__()
        inter_ch = inter_ch or skip_ch // 2
        self.W_gate = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.W_skip = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, 1, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, gate, skip):
        g = F.interpolate(self.W_gate(gate), size=skip.shape[2:],
                          mode="bilinear", align_corners=False)
        s = self.W_skip(skip)
        attn = self.psi(g + s)          # (B, 1, H, W)
        return skip * attn


class UpBlock(nn.Module):
    """
    Decoder block: bilinear upsample → attention gate → concat → ResDoubleConv.

    Replaces your ConvTranspose2d blocks.
    """

    def __init__(self, in_ch, skip_ch, out_ch, use_attention=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up_conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.use_attention = use_attention
        if use_attention:
            self.attn = AttentionGate(gate_ch=out_ch, skip_ch=skip_ch)
        self.conv = ResDoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up_conv(self.up(x))
        # Handle size mismatch from odd dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                              align_corners=False)
        if self.use_attention:
            skip = self.attn(x, skip)
        return self.conv(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    """
    Improved U-Net.

    Parameters
    ----------
    in_ch : int       Input channels (1 for grayscale ultrasound).
    num_classes : int  Output classes (5 = background + 4 chambers).
    base_ch : int      Base channels. 64 recommended (was 32 in your code,
                       but you already use 64 in training — good).
    use_attention : bool  Attention gates in decoder.
    deep_supervision : bool  Auxiliary loss heads at mid-decoder levels.
    dropout : float    Dropout in bottleneck.

    Architecture (base_ch=64, input 384×384):
        enc1:  64  @ 384×384
        enc2: 128  @ 192×192
        enc3: 256  @  96×96
        enc4: 512  @  48×48     ← NEW (was bottleneck)
        bottleneck: 1024 @ 24×24  ← NEW
    """

    def __init__(self, in_ch=1, num_classes=5, base_ch=64,
                 use_attention=True, deep_supervision=True, dropout=0.15):
        super().__init__()
        self.deep_supervision = deep_supervision
        c = base_ch

        # ---- Encoder ----
        self.enc1 = ResDoubleConv(in_ch, c)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ResDoubleConv(c, c * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ResDoubleConv(c * 2, c * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ResDoubleConv(c * 4, c * 8)          # NEW level
        self.pool4 = nn.MaxPool2d(2)

        # ---- Bottleneck ----
        self.bottleneck = ResDoubleConv(c * 8, c * 16, dropout=dropout)

        # ---- Decoder ----
        self.up4 = UpBlock(c * 16, c * 8, c * 8, use_attention)
        self.up3 = UpBlock(c * 8,  c * 4, c * 4, use_attention)
        self.up2 = UpBlock(c * 4,  c * 2, c * 2, use_attention)
        self.up1 = UpBlock(c * 2,  c,     c,     use_attention)

        # ---- Output ----
        self.out_conv = nn.Conv2d(c, num_classes, 1)

        # ---- Deep supervision heads ----
        if deep_supervision:
            self.ds3 = nn.Conv2d(c * 4, num_classes, 1)  # @ H/4
            self.ds2 = nn.Conv2d(c * 2, num_classes, 1)  # @ H/2

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        H, W = x.shape[2:]

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        bn = self.bottleneck(self.pool4(e4))

        # Decoder
        d4 = self.up4(bn, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        logits = self.out_conv(d1)

        if self.deep_supervision and self.training:
            ds3_out = F.interpolate(self.ds3(d3), (H, W), mode="bilinear",
                                     align_corners=False)
            ds2_out = F.interpolate(self.ds2(d2), (H, W), mode="bilinear",
                                     align_corners=False)
            return logits, [ds2_out, ds3_out]

        return logits
