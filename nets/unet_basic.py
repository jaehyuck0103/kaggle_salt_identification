import torch.nn as nn

from nets.unet_basic_parts import DoubleConv, Down, Up


class UNetBasic(nn.Module):
    def __init__(self, cfg):
        super(UNetBasic, self).__init__()
        CH = cfg.NET_CH
        self.in_block = DoubleConv(1, CH*1)
        self.down1 = Down(CH*1, CH*2)
        self.down2 = Down(CH*2, CH*4)
        self.down3 = Down(CH*4, CH*8)
        self.down4 = Down(CH*8, CH*16)
        self.up1 = Up(CH*16, CH*8)
        self.up2 = Up(CH*8, CH*4)
        self.up3 = Up(CH*4, CH*2)
        self.up4 = Up(CH*2, CH*1)
        self.out_block = nn.Conv2d(CH*1, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.in_block(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out_block(x)
        return x
