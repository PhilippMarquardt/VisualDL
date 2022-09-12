from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import modules as md
import torch.nn.functional as F
from torch.nn import *
from torch import nn
import torch


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relu = ReLU()
        self.block1_conv1 = Conv2d(in_channels, out_channels, 1)
        self.block1_bn = BatchNorm2d(out_channels)

        self.block2_conv = Conv2d(in_channels, out_channels, 1, dilation=1, bias=False)
        self.block2_bn = BatchNorm2d(out_channels)

        self.block3_conv = Conv2d(in_channels, out_channels, 3, dilation=6, bias=False)
        self.block3_bn = BatchNorm2d(out_channels)

        self.block4_conv = Conv2d(in_channels, out_channels, 3, dilation=12, bias=False)
        self.block4_bn = BatchNorm2d(out_channels)

        self.block5_conv = Conv2d(in_channels, out_channels, 3, dilation=18, bias=False)
        self.block5_bn = BatchNorm2d(out_channels)

        self.conv_out = Conv2d(
            out_channels * 5, out_channels, 1, dilation=1, bias=False
        )
        self.bn_out = BatchNorm2d(out_channels)

    def forward(self, x):
        shape = x.shape
        y1 = AvgPool2d((shape[2], shape[3]))(x)
        y1 = self.block1_conv1(y1)
        y1 = self.block1_bn(y1)
        y1 = self.relu(y1)
        y1 = Upsample((shape[2], shape[3]))(y1)

        y2 = self.block2_conv(x)
        y2 = self.block2_bn(y2)
        y2 = self.relu(y2)

        y3 = self.block2_conv(x)
        y3 = self.block2_bn(y3)
        y3 = self.relu(y3)

        y4 = self.block2_conv(x)
        y4 = self.block2_bn(y4)
        y4 = self.relu(y4)

        y5 = self.block2_conv(x)
        y5 = self.block2_bn(y5)
        y5 = self.relu(y5)

        y = torch.cat([y1, y2, y3, y4, y5], axis=1)
        y = self.conv_out(y)
        y = self.bn_out(y)
        y = self.relu(y)
        return y


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderBlockDouble(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        in_channels2,
        use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels + in_channels2,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x, skip=None, skip2=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        if skip2 is not None:
            x = torch.cat([x, skip2], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DoubleUnet(nn.Module):
    def __init__(self, encoder_name="resnet34", classes=1):
        super().__init__()
        self.classes = classes
        self.encoder1 = get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=None,
        )
        self.encoder2 = get_encoder(
            encoder_name,
            in_channels=3,
            depth=5,
            weights=None,
        )

        # DECODER 1
        decoder_channels = (256, 128, 64, 32, 16)
        encoder_channels = self.encoder1.out_channels[1:]
        encoder_channels = encoder_channels[::-1]
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.center = Identity()
        ##################################################
        # DECODER 2
        decoder_channels = (256, 128, 64, 32, 16)
        encoder_channels = self.encoder2.out_channels[1:]
        encoder_channels = encoder_channels[::-1]
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels
        blocks = [
            DecoderBlockDouble(in_ch, skip_ch, out_ch, skip_ch)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks2 = nn.ModuleList(blocks)
        ##################################################
        self.aspp1 = ASPP(
            self.encoder1.out_channels[-1], self.encoder1.out_channels[-1]
        )
        self.aspp2 = ASPP(
            self.encoder1.out_channels[-1], self.encoder1.out_channels[-1]
        )
        self.out_class_left = torch.nn.Conv2d(decoder_channels[-1], classes, 1)
        self.out = torch.nn.Conv2d(decoder_channels[-1], 1, 1)

        self.relu = ReLU()
        self.sigmoid = Sigmoid()
        self.out_class_right = torch.nn.Conv2d(decoder_channels[-1], classes, 1)

        self.final_out = torch.nn.Conv2d(classes * 2, classes, 1)

    def forward(self, x):
        inp = x.clone()
        features = self.encoder1(x)
        x = self.aspp1(features[-1])

        # DECODER 1
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips_first = features[1:]

        x = self.center(x)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips_first[i] if i < len(skips_first) else None
            x = decoder_block(x, skip)
        ################################################

        multiply_out = self.relu(self.out_class_left(x))

        left_out = self.sigmoid(self.out(x))

        multiply = inp * left_out

        features = self.encoder2(multiply)
        x = self.aspp2(features[-1])

        # DECODER 2
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(x)
        for i, decoder_block in enumerate(self.blocks2):
            skip = skips[i] if i < len(skips) else None
            skip_2 = skips_first[i] if i < len(skips_first) else None
            x = decoder_block(x, skip, skip_2)
        #################################################

        x = self.out_class_right(x)
        self.final_out(torch.cat([multiply_out, x], axis=1))
        return x
