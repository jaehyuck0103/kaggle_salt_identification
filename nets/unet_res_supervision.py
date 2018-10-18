from torch import nn
from torch.nn import functional as F
import torch

from .resnet import resnet34
from .cbam import CBAM


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                            kernel_size, padding=kernel_size//2),
                                  nn.BatchNorm2d(out_channels),
                                  CBAM(out_channels),
                                  nn.ReLU(inplace=True)
                                  )

    def forward(self, x):
        return self.conv(x)


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv):
        super(DecoderBlockV2, self).__init__()

        if is_deconv:
            self.upsample = nn.Sequential(
                ConvBnRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4,
                                   stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.upsample = nn.Sequential(
                ConvBnRelu(in_channels, out_channels),
                nn.Upsample(scale_factor=2, mode='bilinear'),
            )

    def forward(self, x):
        x = self.upsample(x)
        return x


class UNetResSupervision(nn.Module):

    def __init__(self, dropout=0.5, pretrained=True, is_deconv=False):
        super().__init__()

        self.encoder = resnet34(pretrained=pretrained)

        self.enc0 = nn.Sequential(self.encoder.conv1,
                                  self.encoder.bn1,
                                  self.encoder.relu)
        self.maxpool = self.encoder.maxpool

        self.enc1 = self.encoder.layer1
        self.enc2 = self.encoder.layer2
        self.enc3 = self.encoder.layer3
        self.enc4 = self.encoder.layer4

        self.squ0 = ConvBnRelu(64, 32, kernel_size=1)
        self.squ1 = ConvBnRelu(64, 32, kernel_size=1)
        self.squ2 = ConvBnRelu(128, 32, kernel_size=1)
        self.squ3 = ConvBnRelu(256, 32, kernel_size=1)
        self.squ4 = ConvBnRelu(512, 64, kernel_size=1)

        self.dec4 = DecoderBlockV2(64, 64, 32, is_deconv)
        self.dec3 = DecoderBlockV2(32 + 32, 64, 32, is_deconv)
        self.dec2 = DecoderBlockV2(32 + 32, 64, 32, is_deconv)
        self.dec1 = DecoderBlockV2(32 + 32, 64, 32, is_deconv)
        self.dec0 = ConvBnRelu(32 + 32, 32)

        self.hyper = nn.Sequential(
            ConvBnRelu(224, 64),
            nn.Dropout(p=dropout),
        )

        self.final_seg_pure = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        # for imgae classification
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Dropout(p=dropout),
            # ConvBnRelu(512, 64, kernel_size=1),
            nn.Conv2d(512, 64, kernel_size=1),
            nn.ReLU(inplace=True),
        )

        self.final_cls = nn.Linear(64, 1)

        # fuse
        self.final_fuse = nn.Conv2d(128, 1, kernel_size=3, padding=1)

        '''
        print(sum(p.numel() for p in self.input_adjust.parameters() if p.requires_grad))
        print(sum(p.numel() for p in self.enc1.parameters() if p.requires_grad))
        print(sum(p.numel() for p in self.enc2.parameters() if p.requires_grad))
        print(sum(p.numel() for p in self.enc3.parameters() if p.requires_grad))
        print(sum(p.numel() for p in self.enc4.parameters() if p.requires_grad))
        exit()
        '''
        '''
        print(sum(p.numel() for p in self.dec4.parameters() if p.requires_grad))
        print(sum(p.numel() for p in self.dec3.parameters() if p.requires_grad))
        print(sum(p.numel() for p in self.dec2.parameters() if p.requires_grad))
        print(sum(p.numel() for p in self.dec1.parameters() if p.requires_grad))
        print(sum(p.numel() for p in self.final.parameters() if p.requires_grad))
        exit()
        '''

    def forward(self, x):
        enc0 = self.enc0(x)  # 64x128x128
        enc0_pool = self.maxpool(enc0)  # 64x64x64
        enc1 = self.enc1(enc0_pool)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        squ0 = self.squ0(enc0)
        squ1 = self.squ1(enc1)
        squ2 = self.squ2(enc2)
        squ3 = self.squ3(enc3)
        squ4 = self.squ4(enc4)

        dec4 = self.dec4(squ4)
        dec3 = self.dec3(torch.cat([dec4, squ3], 1))
        dec2 = self.dec2(torch.cat([dec3, squ2], 1))
        dec1 = self.dec1(torch.cat([dec2, squ1], 1))
        dec0 = self.dec0(torch.cat([dec1, squ0], 1))

        # hypercolumn
        hyper = torch.cat((
            dec0,
            dec1,
            F.interpolate(dec2, scale_factor=2, mode='bilinear'),
            F.interpolate(dec3, scale_factor=4, mode='bilinear'),
            F.interpolate(dec4, scale_factor=8, mode='bilinear'),
            F.interpolate(squ4, scale_factor=16, mode='bilinear'),
        ), 1)
        hyper = self.hyper(hyper)

        final_seg_pure = self.final_seg_pure(hyper)

        # cls
        avg_pool = self.avg_pool(enc4)
        avg_pool_sq = avg_pool.squeeze(dim=3)
        avg_pool_sq = avg_pool_sq.squeeze(dim=2)  # batch가 1인 경우 때문에 squeeze 두번에 걸쳐서.
        final_cls = self.final_cls(avg_pool_sq)

        # fuse
        fuse = torch.cat((
            hyper,
            F.interpolate(avg_pool, scale_factor=128, mode='nearest'),
        ), 1)
        final_fuse = self.final_fuse(fuse)

        return final_fuse, final_seg_pure, final_cls
