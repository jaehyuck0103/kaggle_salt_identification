from torch import nn
from torch.nn import functional as F
import torch
import torchvision


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                            kernel_size, padding=kernel_size//2),
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

        self.upsample = nn.Sequential(
            ConvBnRelu(in_channels, out_channels),
            nn.Upsample(scale_factor=2, mode='bilinear'),
        )

    def forward(self, x):
        if self.is_deconv:
            x = self.deconv(x)
        else:
            x = self.upsample(x)
        return x


class UNetResOpen(nn.Module):
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
            bottom_channel_nr = 512
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        self.input_adjust = nn.Sequential(self.encoder.conv1,
                                          self.encoder.bn1,
                                          self.encoder.relu)

        self.enc1 = self.encoder.layer1
        self.enc2 = self.encoder.layer2
        self.enc3 = self.encoder.layer3
        self.enc4 = self.encoder.layer4

        self.squ1 = ConvBnRelu(64, 32, kernel_size=1)
        self.squ2 = ConvBnRelu(128, 32, kernel_size=1)
        self.squ3 = ConvBnRelu(256, 32, kernel_size=1)
        self.squ4 = ConvBnRelu(512, 64, kernel_size=1)

        self.dec4 = DecoderBlockV2(64, num_filters * 16,
                                   num_filters, is_deconv)
        self.dec3 = DecoderBlockV2(32 + num_filters, num_filters * 8,
                                   num_filters, is_deconv)
        self.dec2 = DecoderBlockV2(32 + num_filters, num_filters * 4,
                                   num_filters, is_deconv)
        self.dec1 = DecoderBlockV2(32 + num_filters, num_filters * 2,
                                   num_filters, is_deconv)
        self.final = nn.Sequential(
            ConvBnRelu(192, num_filters * 2),
            nn.Conv2d(num_filters * 2, num_classes, kernel_size=1),
        )

        '''
        print(sum(p.numel() for p in self.dec4.parameters() if p.requires_grad))
        print(sum(p.numel() for p in self.dec3.parameters() if p.requires_grad))
        print(sum(p.numel() for p in self.dec2.parameters() if p.requires_grad))
        print(sum(p.numel() for p in self.dec1.parameters() if p.requires_grad))
        print(sum(p.numel() for p in self.final.parameters() if p.requires_grad))
        exit()
        '''

    def forward(self, x):
        input_adjust = self.input_adjust(x)
        enc1 = self.enc1(input_adjust)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        squ1 = self.squ1(enc1)
        squ2 = self.squ2(enc2)
        squ3 = self.squ3(enc3)
        squ4 = self.squ4(enc4)

        dec4 = self.dec4(squ4)
        dec3 = self.dec3(torch.cat([dec4, squ3], 1))
        dec2 = self.dec2(torch.cat([dec3, squ2], 1))
        dec1 = self.dec1(torch.cat([dec2, squ1], 1))

        # hypercolumn
        y = torch.cat((
            dec1,
            F.interpolate(dec2, scale_factor=2, mode='bilinear'),
            F.interpolate(dec3, scale_factor=4, mode='bilinear'),
            F.interpolate(dec4, scale_factor=8, mode='bilinear'),
            F.interpolate(squ4, scale_factor=16, mode='bilinear'),
        ), 1)

        y = F.dropout2d(y, p=self.dropout_2d)
        y = self.final(y)
        return y
