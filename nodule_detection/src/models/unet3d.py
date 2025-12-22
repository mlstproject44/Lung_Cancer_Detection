import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    #this is the basic building block, two 2d convolutions with normalization, ReLU and dropout
    def __init__(  #channels - different feature maps or layers of information at each spatial location
        self,
        input_channels: int,
        output_channels: int,
        dropout: float = 0.0,
        residual: bool = False  #ResNet skip connections
    ): 
        super().__init__()
        self.residual = residual

        num_groups = min(8, output_channels) 
        norm_layer1 = nn.GroupNorm(num_groups, output_channels)
        norm_layer2 = nn.GroupNorm(num_groups, output_channels)

        layers = [  #first convolution block
            nn.Conv3d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            norm_layer1,  #normalization layer
            nn.ReLU(inplace=True)
        ]
        if dropout > 0:
            layers.append(nn.Dropout3d(p=dropout))  #dropout layer

        layers.extend([
            nn.Conv3d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            norm_layer2,
            nn.ReLU(inplace=True)
        ])
        if dropout > 0:
            layers.append(nn.Dropout3d(p=dropout))

        self.double_conv = nn.Sequential(*layers)  #creates a container that runs layers in order

        #if in_channels != out_channels, we need a 1x1 conv to match dimensions
        if residual and input_channels != output_channels:  
            self.residual_proj = nn.Conv3d(input_channels, output_channels, kernel_size=1, bias=False)
        else:
            self.residual_proj = None
    
    def forward(self, x):
        #forward pass through the double convolution block
        output = self.double_conv(x)  #x: input tensor of shape(batch, input_channels, depth, height, width)

        if self.residual:
            if self.residual_proj is not None:
                x = self.residual_proj(x)  #projects input to match output dimensions
            output = output + x
        
        return output

class Down(nn.Module):
    #downsampling block for encoder, reduces spatial dimensions by half while increasing feature channels
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dropout: float = 0.0,
        residual: bool = False
    ):
        super().__init__()

        self.maxpool = nn.MaxPool3d(2)  #maxpooling layer reduces spatial dimensions by factor of 2
        self.conv = DoubleConv(input_channels, output_channels, dropout, residual)
    
    def forward(self, x):
        x = self.maxpool(x)
        return self.conv(x)

class AttentionGate(nn.Module):
    #attention mechanism helps model to focus on relevant regions by learning to suppress 
    #irrelevant activations in the skip connections
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int):
        super().__init__()

        self.W_g = nn.Sequential(  #transforms gating signal (from decoder) to intermediate representation
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm3d(inter_channels)
        )
        self.W_x = nn.Sequential(  #transforms skip connections (encoder) to intermediate representation
            nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=True),
            nn.BatchNorm3d(inter_channels)
        )
        self.psi = nn.Sequential(  #generates attention map, a heatmap indicating importance of each spatial location
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()  #ensures attention coefficients are 0-1
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, gate, skip):
        #computes attention weighted skip connections

        g1 = self.W_g(gate)  #gating signal - what we're looking for
        x1 = self.W_x(skip)  #skip connection - what's available
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  #attention coefficients

        return skip * psi  #applying attention = amplifying relevant regions

class Up(nn.Module):
    #upsampling block for decoder, increases spatial dimensions while decreasing feature channels
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dropout: float = 0.0,
        residual: bool = False,
        attention: bool = False
    ):
        super().__init__()

        #transposed convolution for upsampling
        self.up = nn.ConvTranspose3d(input_channels, input_channels // 2, kernel_size=2, stride=2)
        #double convolution after concatenating skip connections and upsampled features
        self.conv = DoubleConv(input_channels, output_channels, dropout, residual)

        self.use_attention = attention
        if attention:
            self.attention_gate = AttentionGate(
                gate_channels=input_channels // 2,  #channels from upsampled decoder
                skip_channels=input_channels // 2,  #channels from encoder skip connection
                inter_channels=output_channels  #intermediate channels for attention
            )
    
    def forward(self, x1, x2):
        x1 = self.up(x1)  #upsample decoder features by 2x
        if self.use_attention:
            x2 = self.attention_gate(gate=x1, skip=x2)

        x = torch.cat([x2, x1], dim=1)  #concatenate skip connection with upsampled features along channel dimension
        return self.conv(x)
    
class UNet3D(nn.Module):
    #3d U-Net architecture for volumetric segmentation
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        init_features: int = 32,  #number of feature channels in first encoder layer
        dropout: float = 0.15,
        checkpointing: bool = False
    ):
        super().__init__()

        self.checkpointing = checkpointing  #gradient checkpoint - trade computation for memory
        features = init_features
        residual = True
        attention = True

        #======== ENCODER ========
        #progressively downsamples spatial dimensions while increasing channel depth
        self.encoder1 = DoubleConv(input_channels, features, dropout, residual)
        self.encoder2 = Down(features, features * 2, dropout, residual)
        self.encoder3 = Down(features * 2, features * 4, dropout, residual)
        self.encoder4 = Down(features * 4, features * 8, dropout, residual)

        self.bottleneck = Down(features * 8, features * 16, dropout, residual)  #bottleneck

        #======== DECODER ========
        #progressively upsamples spatial dimensions while decreasing channel depth
        self.decoder4 = Up(features * 16, features * 8, dropout, residual, attention)
        self.decoder3 = Up(features * 8, features * 4, dropout, residual, attention)
        self.decoder2 = Up(features * 4, features * 2, dropout, residual, attention)
        self.decoder1 = Up(features * 2, features, dropout, residual, attention)

        self.output_conv = nn.Conv3d(features, output_channels, kernel_size=1)  #outputs raw logits
    
    def _forward_implementation(self, x):
        #internal forward pass implementation
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)

        dec4 = self.decoder4(bottleneck, enc4)  #upsample and merge with enc4
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)

        output = self.output_conv(dec1)  #final 1x1 convolution to produce segmentation map
        return output
    
    def forward(self, x):
        #forward pass through U-Net
        if self.checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward_implementation, x, use_reentrant=False)
        else:
            return self._forward_implementation(x)