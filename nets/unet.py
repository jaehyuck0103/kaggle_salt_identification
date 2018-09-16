import torch.nn as nn

from nets.unet_parts import inconv, down, up, outconv


# 32
class UNet(nn.Module):
    def __init__(self, cfg):
        super(UNet, self).__init__()
        CH = cfg.NET_CH
        self.inc = inconv(1, CH*1)
        self.down1 = down(CH*1, CH*2)
        self.down2 = down(CH*2, CH*4)
        self.down3 = down(CH*4, CH*8)
        self.down4 = down(CH*8, CH*16)
        self.up1 = up(CH*16, CH*8)
        self.up2 = up(CH*8, CH*4, out_pad=1)
        self.up3 = up(CH*4, CH*2)
        self.up4 = up(CH*2, CH*1, out_pad=1)
        self.outc = outconv(CH*1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
