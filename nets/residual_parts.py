import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.block(x)
        y += x
        y = self.act(y)
        return y


class DoubleResidual(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleResidual, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            ResidualBlock(out_ch),
            ResidualBlock(out_ch),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate):
        super(Down, self).__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            # nn.Dropout2d(dropout_rate),
            DoubleResidual(in_ch, out_ch),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate, pad=0, out_pad=0):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2,
                                     padding=pad, output_padding=out_pad)

        self.block = nn.Sequential(
            # nn.Dropout2d(dropout_rate),
            DoubleResidual(in_ch, out_ch),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.block(x)
        return x
