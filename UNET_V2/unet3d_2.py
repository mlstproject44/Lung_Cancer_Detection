import torch
import torch.nn as nn
import math


class DoubleConv(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dropout: float = 0.0,
        residual: bool = False
    ):
        super().__init__()
        self.residual = residual

        num_groups = min(8, output_channels)
        norm_layer1 = nn.GroupNorm(num_groups, output_channels)
        norm_layer2 = nn.GroupNorm(num_groups, output_channels)

        layers = [
            nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            norm_layer1,
            nn.ReLU(inplace=True)
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(p=dropout))

        layers.extend([
            nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            norm_layer2,
            nn.ReLU(inplace=True)
        ])
        if dropout > 0:
            layers.append(nn.Dropout3d(p=dropout))

        self.double_conv = nn.Sequential(*layers)

        if residual and input_channels != output_channels:
            self.residual_proj = nn.Conv3d(input_channels, output_channels, kernel_size=1, bias=False)
        else:
            self.residual_proj = None

    def forward(self, x):
        output = self.double_conv(x)
        if self.residual:
            if self.residual_proj is not None:
                x = self.residual_proj(x)
            output = output + x
        return output


class Down(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dropout: float = 0.0,
        residual: bool = False
    ):
        super().__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.conv = DoubleConv(input_channels, output_channels, dropout, residual)

    def forward(self, x):
        x = self.maxpool(x)
        return self.conv(x)


# ============================================================
#  COVARIANCE ATTENTION GATE
# ============================================================
class CovarianceAttentionGate(nn.Module):
    """
    Cross-covariance channel attention gate.

    For each skip channel, computes how much it co-varies with the
    decoder's gating signal. Channels that statistically co-vary with
    "what the decoder is currently searching for" get amplified.
    This is a second-order generalisation of additive attention.

    The bmm producing the (B, Cs, Cg) cross-covariance is done in fp32
    regardless of AMP, because at decoder4 Cs=Cg=384 and summing 1728
    spatial positions in fp16 overflows to NaN.

    A learnable per-channel importance scalar (C,) in [0,1] is also
    maintained so the training loop can pass it to ChannelAwareFocalLoss.
    """
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()

        # normalisation before cross-covariance (keeps activations stable)
        self.norm_g = nn.GroupNorm(min(8, gate_channels), gate_channels)
        self.norm_s = nn.GroupNorm(min(8, skip_channels), skip_channels)

        # learnable per-channel importance — initialised at 0 → sigmoid = 0.5
        self.channel_importance_raw = nn.Parameter(torch.zeros(skip_channels))

        # 1×1 conv mixes channels after attention weighting
        self.proj = nn.Conv3d(skip_channels, skip_channels, kernel_size=1, bias=True)

    @property
    def channel_importance(self) -> torch.Tensor:
        return torch.sigmoid(self.channel_importance_raw)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        B, Cg, D, H, W = gate.shape
        B, Cs, _, _, _ = skip.shape
        N = D * H * W

        g = self.norm_g(gate).view(B, Cg, N)
        s = self.norm_s(skip).view(B, Cs, N)

        with torch.autocast(device_type=gate.device.type, enabled=False):
            # SCALE BEFORE BMM: Use math.sqrt(N) to keep the dot product in a sane range
            # This prevents the 'NaN' during the internal sum of bmm
            scaling_factor = math.sqrt(N)
            s_f = s.float() / scaling_factor
            g_f = g.float() / scaling_factor

            cross_cov = torch.bmm(s_f, g_f.transpose(1, 2))  # (B, Cs, Cg)

        # Sigmoid keeps each channel independently in (0, 1), so the max
        # multiplier at init is 1 * 1.5 = 1.5 — safe under AMP/fp16.
        # Previously used softmax(...) * Cs which could produce values ~384
        # at decoder4, causing NaN in fp16.
        relevance = torch.sigmoid(cross_cov.abs().mean(dim=2))  # (B, Cs), each in (0, 1)

        relevance = relevance.view(B, Cs, 1, 1, 1).to(skip.dtype)
        imp = self.channel_importance.view(1, Cs, 1, 1, 1)

        return self.proj(skip * (relevance * (1.0 + imp)))


# ============================================================
#  UP BLOCK
# ============================================================
class Up(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dropout: float = 0.0,
        residual: bool = False,
        attention: bool = False
    ):
        super().__init__()

        self.up = nn.ConvTranspose3d(input_channels, input_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(input_channels, output_channels, dropout, residual)

        self.use_attention = attention
        if attention:
            self.attention_gate = CovarianceAttentionGate(
                gate_channels=input_channels // 2,
                skip_channels=input_channels // 2,
                inter_channels=output_channels
            )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        if self.use_attention:
            x2 = self.attention_gate(gate=x1, skip=x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# ============================================================
#  3-D U-NET
# ============================================================
class UNet3D(nn.Module):
    """
    3-D U-Net with:
      - Residual DoubleConv blocks
      - Cross-covariance attention gates on all skip connections
      - Optional gradient checkpointing
    """
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        init_features: int = 48,
        dropout: float = 0.2,
        checkpointing: bool = False
    ):
        super().__init__()

        self.checkpointing = checkpointing
        features = init_features
        residual = True
        attention = True

        # ======== ENCODER ========
        self.encoder1 = DoubleConv(input_channels, features, dropout, residual)
        self.encoder2 = Down(features, features * 2, dropout, residual)
        self.encoder3 = Down(features * 2, features * 4, dropout, residual)
        self.encoder4 = Down(features * 4, features * 8, dropout, residual)

        self.bottleneck = Down(features * 8, features * 16, dropout, residual)

        # ======== DECODER ========
        self.decoder4 = Up(features * 16, features * 8, dropout, residual, attention)
        self.decoder3 = Up(features * 8,  features * 4, dropout, residual, attention)
        self.decoder2 = Up(features * 4,  features * 2, dropout, residual, attention)
        self.decoder1 = Up(features * 2,  features,     dropout, residual, attention)

        self.output_conv = nn.Conv3d(features, output_channels, kernel_size=1)

    def get_all_channel_importances(self) -> list:
        """
        Returns list of channel_importance tensors from all 4 attention gates.
        Used by ChannelAwareFocalLoss in the training loop.
        """
        return [
            self.decoder4.attention_gate.channel_importance,
            self.decoder3.attention_gate.channel_importance,
            self.decoder2.attention_gate.channel_importance,
            self.decoder1.attention_gate.channel_importance,
        ]

    def _forward_implementation(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)

        dec4 = self.decoder4(bottleneck, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)

        return self.output_conv(dec1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_implementation, x, use_reentrant=False
            )
        return self._forward_implementation(x)