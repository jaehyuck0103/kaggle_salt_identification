import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_ch, out_ch, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(in_ch, out_ch, stride),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            conv3x3(out_ch, out_ch),
            nn.BatchNorm2d(out_ch),
        )

        if in_ch == out_ch and stride == 1:
            self.shortcut = (lambda x: x)  # skip connection
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        x = self.block(x) + self.shortcut(x)
        x = F.relu(x, inplace=True)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, n_blocks, pad=0, out_pad=0):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2,
                                     padding=pad, output_padding=out_pad)

        layers = [BasicBlock(in_ch, out_ch)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_ch, out_ch))

        self.block = nn.Sequential(*layers)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.block(x)
        return x


class UNetRes34(nn.Module):

    def __init__(self, cfg):
        super(UNetRes34, self).__init__()
        CH = cfg.NET_CH
        self.in_block = nn.Sequential(
            nn.Conv2d(1, CH*1, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(CH*1),
            nn.ReLU(inplace=True),
        )
        self.inblock = self._make_layer(1, CH*1, 3)
        self.down1 = self._make_layer(CH*1, CH*2, 3, stride=2)
        self.down2 = self._make_layer(CH*2, CH*4, 4, stride=2)
        self.down3 = self._make_layer(CH*4, CH*8, 6, stride=2)
        self.down4 = self._make_layer(CH*8, CH*16, 3, stride=2)
        self.up1 = Up(CH*16, CH*8, 3)
        self.up2 = Up(CH*8, CH*4, 6)
        self.up3 = Up(CH*4, CH*2, 4)
        self.up4 = Up(CH*2, CH*1, 3)
        self.out_block = nn.Conv2d(CH*1, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_ch, out_ch, n_blocks, stride=1):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_ch, out_ch))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.in_block(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out_block(x)
        return x
