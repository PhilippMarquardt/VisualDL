from .tnt import TNT
from .vit import ViT
from .axial import AxialImageTransformer, AxialAttention
from .custom import Conv2dReLU
import torch
from torch import nn
from torch.nn import functional as F

class AxialBottleneck(nn.Module):
    def __init__(self, dim, heads, image_size):
        super().__init__()
        self.ax = AxialAttention(dim=dim, dim_index=1,axial_pos_emb_shape = (image_size,image_size), heads = heads)

    def forward(self, x):
        tmp = self.ax(x)
        return tmp + x

class DecoderBlock(nn.Module):
    def __init__(self,
                in_channels,
                skip_channels,
                out_channels,
                size,
                depth = 1,
                heads= 8):
        super().__init__()   
        #self.ax = AxialImageTransformer(dim = in_channels + skip_channels,heads= heads,depth = depth,axial_pos_emb_shape = (size,size), reversible=False)
        self.ax = nn.ModuleList([AxialBottleneck(dim=in_channels + skip_channels,heads=heads, image_size=size) for _ in range(depth)])
        self.out = Conv2dReLU(in_channels + skip_channels, out_channels, 1)
        
    def forward(self, x, skip = None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.out(self.ax(x))


class UnetDecoderBlock(nn.Module):
    def __init__(self,
                in_channels,
                skip_channels,
                out_channels,
                size,
                depth = 1,
                heads= 8):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1)

        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class TransInUnet(nn.Module):
    def __init__(self, image_size, nc, init_dim = 16):
        super().__init__()
        self.first_conv = Conv2dReLU(3, init_dim, 1)

        #self.init_ax = AxialImageTransformer(dim = init_dim,depth = init_dim,axial_pos_emb_shape = (image_size,image_size))
        self.init_ax = AxialAttention(dim=init_dim, dim_index=1)
        self.init_conv = Conv2dReLU(init_dim, init_dim, kernel_size=3,padding=1)
            

        self.last_conv_before = Conv2dReLU(init_dim, init_dim, kernel_size=3,padding=1)
        self.last_conv = Conv2dReLU(init_dim, nc, 1)
        # self.first = TNT(image_size = image_size, patch_size = 2, pixel_dim = 16, pixel_size = 1, patch_dim = init_dim * 2, depth = 2, heads = 4, channels = init_dim)
        # self.second = TNT(image_size = image_size//2, patch_size = 2, pixel_dim = 16, pixel_size = 1, patch_dim = init_dim * 4, depth = 4, heads = 6, channels = init_dim * 2)
        # self.third = TNT(image_size = image_size//4, patch_size = 2, pixel_dim = 16, pixel_size = 1, patch_dim = init_dim * 8, depth = 6, heads = 8, channels = init_dim * 4)
        # self.fourth = TNT(image_size = image_size//8, patch_size = 2, pixel_dim = 16, pixel_size = 1, patch_dim = init_dim * 16, depth = 8, heads = 12, channels = init_dim * 8)
        self.first = TNT(image_size = image_size, patch_size = 2, pixel_dim = 16, pixel_size = 1, patch_dim = init_dim * 2, depth = 2, heads = 4, channels = init_dim)
        self.second = TNT(image_size = image_size//2, patch_size = 2, pixel_dim = 16, pixel_size = 1, patch_dim = init_dim * 4, depth = 4, heads = 6, channels = init_dim * 2)
        self.third = TNT(image_size = image_size//4, patch_size = 2, pixel_dim = 16, pixel_size = 1, patch_dim = init_dim * 8, depth = 6, heads = 8, channels = init_dim * 4)
        self.fourth = TNT(image_size = image_size//8, patch_size = 2, pixel_dim = 16, pixel_size = 1, patch_dim = init_dim * 16, depth = 8, heads = 12, channels = init_dim * 8)
        self.cup = ViT(
            image_size = int(image_size / (2**(4 - 1))),
            patch_size = 1,
            dim = init_dim * 32,
            depth = 8,
            heads = 12,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1,
            channels = init_dim * 16
        )

        self.up1 = UnetDecoderBlock(init_dim*32, init_dim*8, init_dim*8, depth = 1, heads = 1,size = image_size//8)
        self.up2 = UnetDecoderBlock(init_dim*8, init_dim*4, init_dim*4, depth = 1, heads = 1,size = image_size//4)
        self.up3 = UnetDecoderBlock(init_dim*4, init_dim*2, init_dim*2, depth = 1, heads = 1,size = image_size//2)
        self.up4 = UnetDecoderBlock(init_dim*2, init_dim, init_dim, depth = 1, heads = 1,size = image_size)

    def forward(self, x):
        #out1 = self.init_ax(self.first_conv(x)) #128
        out1 = self.init_conv(self.first_conv(x)) #128
        out2 = self.first(out1) #64



        out3 = self.second(out2) #32
        out4 = self.third(out3) #16
        out5 = self.fourth(out4) #8
        out6 = self.cup(out5) #8

        x = self.up4(self.up3(self.up2(self.up1(out6, out4), out3), out2), out1)
        return self.last_conv(self.last_conv_before(x))