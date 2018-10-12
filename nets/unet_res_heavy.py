from torch import nn
from torch.nn import functional as F
import torch
import torchvision
from torchvision.models.resnet import BasicBlock


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=True)
                                  )

    def forward(self, x):
        return self.conv(x)


class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlockV2, self).__init__()
        self.is_deconv = is_deconv

        self.deconv = nn.Sequential(
            ConvBnRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        '''
        self.upsample = nn.Sequential(
            ConvBnRelu(in_channels, out_channels),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )
        '''

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ConvBnRelu(in_channels, out_channels),
            BasicBlock(out_channels, out_channels),
        )

    def forward(self, x):
        if self.is_deconv:
            x = self.deconv(x)
        else:
            x = self.upsample(x)
        return x


class UNetResHeavy(nn.Module):
    """PyTorch U-Net model using ResNet(34, 101 or 152) encoder.
    UNet: https://arxiv.org/abs/1505.04597
    ResNet: https://arxiv.org/abs/1512.03385
    Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
    Args:
            encoder_depth (int): Depth of a ResNet encoder (34, 101 or 152).
            num_classes (int): Number of output classes.
            num_filters (int, optional): Number of filters in the last layer of decoder.
            dropout_2d (float, optional): dropout layer before output layer.
            pretrained (bool, optional):
            is_deconv (bool, optional):
    """

    def __init__(self, encoder_depth=34, num_classes=1, num_filters=32, dropout_2d=0.4,
                 pretrained=True, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        if encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        self.input_adjust = nn.Sequential(self.encoder.conv1,
                                          self.encoder.bn1,
                                          self.encoder.relu)

        self.enc1 = self.encoder.layer1  # 64x64x64
        self.enc2 = self.encoder.layer2  # 128x32x32
        self.enc3 = self.encoder.layer3  # 256x16x16
        self.enc4 = self.encoder.layer4  # 512x8x8

        self.center = nn.Sequential(ConvBnRelu(512, 256),
                                    ConvBnRelu(256, 256))

        self.dec4 = DecoderBlockV2(256, 512, 64, is_deconv)
        self.dec3 = DecoderBlockV2(64 + 256, 512, 64, is_deconv)
        self.dec2 = DecoderBlockV2(64 + 128, 256, 64, is_deconv)
        self.dec1 = DecoderBlockV2(64 + 64, 128, 64, is_deconv)
        self.final = nn.Sequential(ConvBnRelu(512, 128),
                                   nn.Conv2d(128, num_classes, kernel_size=1))

    def forward(self, x):
        input_adjust = self.input_adjust(x)
        enc1 = self.enc1(input_adjust)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        dec4 = self.dec4(center)
        dec3 = self.dec3(torch.cat([dec4, enc3], 1))
        dec2 = self.dec2(torch.cat([dec3, enc2], 1))
        dec1 = self.dec1(torch.cat([dec2, enc1], 1))

        # hypercolumn
        y = torch.cat((
            dec1,
            F.interpolate(dec2, scale_factor=2, mode='bilinear'),
            F.interpolate(dec3, scale_factor=4, mode='bilinear'),
            F.interpolate(dec4, scale_factor=8, mode='bilinear'),
            F.interpolate(center, scale_factor=16, mode='bilinear'),
        ), 1)

        y = F.dropout2d(y, p=self.dropout_2d)
        y = self.final(y)
        return y
