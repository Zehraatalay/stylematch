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
    