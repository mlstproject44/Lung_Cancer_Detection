import torch
import torch.nn as nn


class ResidualBlock3D(nn.Module):
    """Basic 3D residual block."""

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else None

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        if self.dropout:
            out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class ResNet3D_18(nn.Module):
    """3D ResNet-18 for binary classification."""

    def __init__(self, in_channels=1, num_classes=1, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        block_dropout = dropout * 0.3
        self.layer1 = self._make_layer(64, 64, 2, stride=1, dropout=block_dropout)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, dropout=block_dropout)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, dropout=block_dropout*1.5)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, dropout=block_dropout*2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1, dropout=0.0):
        layers = [ResidualBlock3D(in_channels, out_channels, stride, dropout)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock3D(out_channels, out_channels, dropout=dropout))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x).squeeze()


class ResNet3D_34(nn.Module):
    """3D ResNet-34 for binary classification."""

    def __init__(self, in_channels=1, num_classes=1, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        block_dropout = dropout * 0.3
        self.layer1 = self._make_layer(64, 64, 3, stride=1, dropout=block_dropout)
        self.layer2 = self._make_layer(64, 128, 4, stride=2, dropout=block_dropout)
        self.layer3 = self._make_layer(128, 256, 6, stride=2, dropout=block_dropout*1.5)
        self.layer4 = self._make_layer(256, 512, 3, stride=2, dropout=block_dropout*2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1, dropout=0.0):
        layers = [ResidualBlock3D(in_channels, out_channels, stride, dropout)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock3D(out_channels, out_channels, dropout=dropout))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x).squeeze()
