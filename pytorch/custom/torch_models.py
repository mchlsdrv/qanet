import logging
import pathlib

import torch
import torch.nn as nn


__author__ = 'sidorov@post.bgu.ac.il'


class RibCage(nn.Module):
    class Conv2DBlk(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size):
            super().__init__()

            self.lyr = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=(1, 1), padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
            )

        def forward(self, x):
            return self.lyr(x)

    class FCBlk(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()

            self.lyr = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=out_features),
                nn.LeakyReLU(0.1, inplace=True),
            )

        def forward(self, x):
            return self.lyr(x)

    def __init__(self, in_channels, out_channels, input_image_shape: tuple, conv2d_out_channels: tuple = (32, 64, 128, 256), conv2d_kernel_sizes: tuple = (5, 5, 5, 5), fc_out_features: tuple = (512, 1024), output_dir: pathlib.Path = pathlib.Path('./output'), logger: logging.Logger = None):
        super().__init__()
        self.in_channels = in_channels
        self.input_image_shape = input_image_shape
        self.out_channels = out_channels
        self.conv2d_out_channels = conv2d_out_channels
        self.conv2d_kernel_sizes = conv2d_kernel_sizes
        self.fc_out_features = fc_out_features
        self.output_dir = output_dir
        self.logger = logger

        self.left_rib_convs = nn.ModuleList()
        self.right_rib_convs = nn.ModuleList()
        self.spine_convs = nn.ModuleList()
        self.fc_lyrs = nn.ModuleList()

        # - Build the conv layers
        self._build_convs()

        # - Build the fc layers
        self.flat_channels = self._get_flat_channels()
        self._build_fcs()

    def _build_convs(self):

        in_chnls = self.in_channels
        for idx, (out_chnls, k_sz) in enumerate(zip(self.conv2d_out_channels, self.conv2d_kernel_sizes)):

            self.left_rib_convs.append(self.Conv2DBlk(in_channels=in_chnls, out_channels=out_chnls, kernel_size=k_sz))

            self.right_rib_convs.append(self.Conv2DBlk(in_channels=in_chnls, out_channels=out_chnls, kernel_size=k_sz))

            self.spine_convs.append(self.Conv2DBlk(in_channels=2*in_chnls if idx == 0 else 3 * in_chnls, out_channels=out_chnls, kernel_size=k_sz))

            in_chnls = out_chnls

    def _build_fcs(self):
        out_feats = self.fc_out_features[0]
        for lyr_idx in range(len(self.fc_out_features)):
            if lyr_idx == 0:
                # - Get flatten layer
                self.fc_lyrs.append(self.FCBlk(in_features=self.flat_channels, out_features=out_feats))
            else:
                in_feats = out_feats
                out_feats = self.fc_out_features[lyr_idx]
                self.fc_lyrs.append(self.FCBlk(in_features=in_feats, out_features=out_feats))

        self.fc_lyrs.append(nn.Linear(in_features=out_feats, out_features=1))

    def convs(self, image, mask):
        x_l = image
        x_r = mask
        for idx, (left_rib, right_rib, spine) in enumerate(zip(self.left_rib_convs, self.right_rib_convs, self.spine_convs)):
            # print(f'x_l.shape: {x_l.shape}, x_r.shape: {x_r.shape}')
            if idx == 0:
                spn = spine(torch.cat((x_l, x_r), dim=1))
            else:
                spn = spine(torch.cat((x_l, x_r, spn), dim=1))
            x_l = left_rib(x_l)
            x_r = right_rib(x_r)

        return torch.cat((x_l, x_r, spn), dim=1)

    def fcs(self, x):
        for fc_lyr in self.fc_lyrs:
            # print(f'x.shape: {x.shape}')
            x = fc_lyr(x)

        return x

    def _get_flat_channels(self):
        rnd_btch = torch.randn(1, self.in_channels, *self.input_image_shape)
        x = self.convs(image=rnd_btch, mask=rnd_btch)

        return x.shape[1] * x.shape[2] * x.shape[3]

    def forward(self, image, mask):
        x = self.convs(image=image, mask=mask).view(-1, self.flat_channels)
        x = self.fcs(x=x)

        return torch.sigmoid(x)
