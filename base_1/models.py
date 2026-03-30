import torch
import torch.nn as nn


# =====================
# Weight Initialization (same as Pix2Pix)
# =====================
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


# =====================
# Generator (basic)
# =====================

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder (Downsampling)
        self.e1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.e2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.e3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.e4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True)
        )

        # Decoder (Upsampling)
        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.d2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.d3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.final = nn.ConvTranspose2d(64, 1, 4, 2, 1)

    def forward(self, x):
        x = self.e1(x)
        x = self.e2(x)
        x = self.e3(x)
        x = self.e4(x)

        x = self.bottleneck(x)

        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)

        out = self.final(x)

        return out  # logits


# =====================
# Discriminator (basic)
# =====================

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_c, out_c, stride=2, use_bn=True):
            layers = [
                nn.Conv2d(in_c, out_c, 4, stride, 1, bias=False)
            ]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(4, 64, use_bn=False),
            block(64, 128),
            block(128, 256),
            block(256, 512)
        )

        # Global classification (NOT patch-based)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1)
        )

    def forward(self, img, mask):
        x = torch.cat([img, mask], dim=1)
        x = self.features(x)
        out = self.classifier(x)
        return out