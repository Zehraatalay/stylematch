import torch
import torch.nn as nn


class SqueezeExcite(nn.Module):

    def __init__(self, channels, reduction=8):
        super().__init__()

        hidden = max(channels // reduction, 4)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        b, c, _, _ = x.shape

        weights = self.pool(x).view(b, c)
        weights = self.fc(weights)
        weights = weights.view(b, c, 1, 1)

        return x * weights


class ResidualSEBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_ch,
                out_ch,
                3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_ch)
        )

        self.se = SqueezeExcite(out_ch)

        if in_ch == out_ch:
            self.shortcut = nn.Identity()

        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    1,
                    bias=False
                ),
                nn.BatchNorm2d(out_ch)
            )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.main(x)
        out = self.se(out)

        return self.act(out + identity)


class MultiScaleStem(nn.Module):

    def __init__(self, out_ch):

        super().__init__()

        branch = out_ch // 2

        self.branch3 = nn.Sequential(
            nn.Conv2d(
                6,
                branch,
                3,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(branch),
            nn.ReLU(inplace=True)
        )

        self.branch5 = nn.Sequential(
            nn.Conv2d(
                6,
                branch,
                5,
                stride=2,
                padding=2,
                bias=False
            ),
            nn.BatchNorm2d(branch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        return torch.cat(
            (
                self.branch3(x),
                self.branch5(x)
            ),
            dim=1
        )
    

class ScratchEarlyFusionNet(nn.Module):
    """Early-fusion compatibility CNN trained without pretrained weights.

    Both RGB images are concatenated along the channel dimension. The model
    outputs one raw compatibility logit per image pair.
    """

    def __init__(self, embed_dim=128, dropout=0.3, base_channels=24):
        super().__init__()
        if base_channels <= 0 or base_channels % 2 != 0:
            raise ValueError("base_channels must be a positive even integer")

        c = base_channels
        self.features = nn.Sequential(
            MultiScaleStem(c),
            nn.MaxPool2d(kernel_size=2),
            ResidualSEBlock(c, 2 * c),
            nn.MaxPool2d(kernel_size=2),
            ResidualSEBlock(2 * c, 4 * c),
            nn.MaxPool2d(kernel_size=2),
            ResidualSEBlock(4 * c, 8 * c),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * c, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, img_a, img_b):
        if img_a.shape != img_b.shape:
            raise ValueError("Both input image tensors must have identical shapes")
        x = torch.cat((img_a, img_b), dim=1)
        x = self.features(x)
        return self.classifier(x).squeeze(1)
