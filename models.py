# Model definitions (Siamese U-Net and GAN components)
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Basic Convolutional Block ---
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), # Bias false with BatchNorm
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

# --- Siamese U-Net Architecture ---
class SiameseUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SiameseUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder Path (Shared Weights)
        self.dconv_down1 = double_conv(n_channels, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)
        self.maxpool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = double_conv(512, 1024)

        # Decoder Path
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # Combining features using absolute difference for skip connections
        self.dconv_up3 = double_conv(1024 + 512, 512) # Input: Upsampled(Bottleneck) + Skip4_diff
        self.dconv_up2 = double_conv(512 + 256, 256)  # Input: Upsampled(Dec4) + Skip3_diff
        self.dconv_up1 = double_conv(256 + 128, 128)  # Input: Upsampled(Dec3) + Skip2_diff
        self.dconv_last = double_conv(128 + 64, 64)     # Input: Upsampled(Dec2) + Skip1_diff

        # Final Layer
        self.conv_last = nn.Conv2d(64, n_classes, 1)

    def forward_encoder(self, x):
        conv1 = self.dconv_down1(x)
        pool1 = self.maxpool(conv1)
        conv2 = self.dconv_down2(pool1)
        pool2 = self.maxpool(conv2)
        conv3 = self.dconv_down3(pool2)
        pool3 = self.maxpool(conv3)
        conv4 = self.dconv_down4(pool3)
        pool4 = self.maxpool(conv4)
        bottleneck = self.bottleneck(pool4)
        return conv1, conv2, conv3, conv4, bottleneck

    def forward(self, x1, x2):
        # --- Encode x1 and x2 --- 
        conv1_1, conv2_1, conv3_1, conv4_1, bottleneck1 = self.forward_encoder(x1)
        conv1_2, conv2_2, conv3_2, conv4_2, bottleneck2 = self.forward_encoder(x2)

        # --- Combine features and Decode ---
        # Upsample bottleneck features (use bottleneck1, could also average or diff)
        up_bottleneck = self.upsample(bottleneck1)

        # Combine skip connection features (absolute difference)
        skip4_diff = torch.abs(conv4_1 - conv4_2)
        x = torch.cat([up_bottleneck, skip4_diff], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)

        skip3_diff = torch.abs(conv3_1 - conv3_2)
        x = torch.cat([x, skip3_diff], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)

        skip2_diff = torch.abs(conv2_1 - conv2_2)
        x = torch.cat([x, skip2_diff], dim=1)
        x = self.dconv_up1(x)
        x = self.upsample(x)

        skip1_diff = torch.abs(conv1_1 - conv1_2)
        x = torch.cat([x, skip1_diff], dim=1)
        x = self.dconv_last(x)

        # Final output
        out = self.conv_last(x)

        # No sigmoid here if using BCEWithLogitsLoss
        # if self.n_classes == 1:
        #     out = torch.sigmoid(out)

        return out

# --- GAN Components (Pix2Pix Style Example) ---

class UNetGenerator(nn.Module):
    """U-Net based Generator (similar to Pix2Pix)."""
    def __init__(self, input_nc, output_nc, num_downs=7, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UNetGenerator, self).__init__()

        # Construct U-Net structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

    def forward(self, input):
        return self.model(input)

# Defines the submodule with skip connection. Ref: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()] # Tanh to map output to [-1, 1]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

import functools # Need this for the UnetSkipConnectionBlock

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator (borrowed from Pix2Pix)."""
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial: # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)] # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


