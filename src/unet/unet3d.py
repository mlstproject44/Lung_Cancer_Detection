import torch
import torch.nn as nn
import torch.utils.checkpoint

class DoubleConv(nn.Module):
    """Two 3D convolutions with BatchNorm, ReLU, and optional Dropout/Residuals."""
    def __init__(self, input_channels: int, output_channels: int, dropout: float = 0.0, residual: bool = False):
        super().__init__()
        self.residual = residual

        num_groups = min(8, output_channels)
        
        layers = [
            nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, output_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(p=dropout))

        layers.extend([
            nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, output_channels),
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
    """Downsampling block (MaxPool + DoubleConv)."""
    def __init__(self, input_channels: int, output_channels: int, dropout: float = 0.0, residual: bool = False):
        super().__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.conv = DoubleConv(input_channels, output_channels, dropout, residual)

    def forward(self, x):
        return self.conv(self.maxpool(x))

class AttentionGate(nn.Module):
    """Attention Gate for filtering skip connection features using BatchNorm3d."""
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm3d(inter_channels)  
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm3d(inter_channels)  
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm3d(1),  
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, gate, skip):
        g = self.W_g(gate)
        x = self.W_x(skip)
        
        psi = self.relu(g + x)
        psi = self.psi(psi)
        
        return skip * psi

class Up(nn.Module):
    """Upsampling block (ConvTranspose + Attention + DoubleConv)."""
    def __init__(self, input_channels: int, output_channels: int, dropout: float = 0.0, residual: bool = False, attention: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose3d(input_channels, input_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(input_channels, output_channels, dropout, residual)
        self.use_attention = attention
        if attention:
            self.attention_gate = AttentionGate(
                gate_channels=input_channels // 2,
                skip_channels=input_channels // 2,
                inter_channels=output_channels
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.use_attention:
            x2 = self.attention_gate(gate=x1, skip=x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet3D(nn.Module):
    """Full 3D Attention U-Net Architecture."""
    def __init__(self, input_channels: int = 1, output_channels: int = 1, init_features: int = 32, dropout: float = 0.2, checkpointing: bool = False):
        super().__init__()
        self.checkpointing = checkpointing
        f = init_features
        res = True
        att = True

        self.encoder1 = DoubleConv(input_channels, f, dropout, res)
        self.encoder2 = Down(f, f * 2, dropout, res)
        self.encoder3 = Down(f * 2, f * 4, dropout, res)
        self.encoder4 = Down(f * 4, f * 8, dropout, res)

        self.bottleneck = Down(f * 8, f * 16, dropout, res)

        self.decoder4 = Up(f * 16, f * 8, dropout, res, att)
        self.decoder3 = Up(f * 8, f * 4, dropout, res, att)
        self.decoder2 = Up(f * 4, f * 2, dropout, res, att)
        self.decoder1 = Up(f * 2, f, dropout, res, att)

        self.output_conv = nn.Conv3d(f, output_channels, kernel_size=1)

    def _forward_implementation(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        bn = self.bottleneck(enc4)

        dec4 = self.decoder4(bn, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)

        return self.output_conv(dec1)

    def forward(self, x):
        if self.checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward_implementation, x, use_reentrant=False)
        return self._forward_implementation(x)