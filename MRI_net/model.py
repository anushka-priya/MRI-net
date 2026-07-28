import torch
import torch.nn as nn

from . import config


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


def make_layer(in_channels, out_channels, num_blocks, stride):
    layers = [BasicBlock(in_channels, out_channels, stride=stride)]
    for _ in range(1, num_blocks):
        layers.append(BasicBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)


class ResNetEncoderFromScratch(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = make_layer(64, 64, 2, stride=1)
        self.layer2 = make_layer(64, 128, 2, stride=2)
        self.layer3 = make_layer(128, 256, 2, stride=2)
        self.layer4 = make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, latent_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_map = self.layer4(x)
        pooled = self.avgpool(feat_map).flatten(1)
        latent = self.fc(pooled)
        return latent, feat_map


class ClassificationHead(nn.Module):
    def __init__(self, latent_dim=256, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, latent):
        return self.net(latent)


class ReconstructionHead(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
        )

    def forward(self, feat_map):
        return self.decoder(feat_map)


class MultiTaskBrainMRIModel(nn.Module):
    def __init__(self, num_classes: int, latent_dim: int = config.LATENT_DIM):
        super().__init__()
        self.encoder = ResNetEncoderFromScratch(latent_dim=latent_dim)
        self.classifier = ClassificationHead(latent_dim=latent_dim, num_classes=num_classes)
        self.reconstructor = ReconstructionHead(in_channels=512)

    def forward(self, x):
        latent, feat_map = self.encoder(x)
        logits = self.classifier(latent)
        recon = self.reconstructor(feat_map)
        return logits, recon


def build_model(num_classes: int, device: torch.device = config.DEVICE) -> MultiTaskBrainMRIModel:
    return MultiTaskBrainMRIModel(num_classes=num_classes).to(device)