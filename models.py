import torch
import torch.nn as nn


# =====================
# Weight Initialization (Pix2Pix standard)
# =====================
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


# =====================
# Generator (UNet)
# =====================

class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c, down=True, use_bn=True):
        super().__init__()

        if down:
            self.block = nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_c) if use_bn else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            layers = [
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            ]

            # Slightly reduced dropout
            if out_c >= 256:
                layers.append(nn.Dropout(0.3))

            self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.d1 = UNetBlock(3, 64, use_bn=False)
        self.d2 = UNetBlock(64, 128)
        self.d3 = UNetBlock(128, 256)
        self.d4 = UNetBlock(256, 512)
        self.d5 = UNetBlock(512, 512)

        # Decoder
        self.u1 = UNetBlock(512, 512, down=False)
        self.u2 = UNetBlock(1024, 256, down=False)
        self.u3 = UNetBlock(512, 128, down=False)
        self.u4 = UNetBlock(256, 64, down=False)

        self.final = nn.ConvTranspose2d(128, 1, 4, 2, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)

        u1 = self.u1(d5)
        u2 = self.u2(torch.cat([u1, d4], 1))
        u3 = self.u3(torch.cat([u2, d3], 1))
        u4 = self.u4(torch.cat([u3, d2], 1))

        out = self.final(torch.cat([u4, d1], 1))

        return out  # logits


# =====================
# Discriminator (PatchGAN)
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

        self.model = nn.Sequential(
            block(4, 64, use_bn=False),
            block(64, 128),
            block(128, 256),
            block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, img, mask):
        x = torch.cat([img, mask], dim=1)
        return self.model(x)
