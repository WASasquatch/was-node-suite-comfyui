# This file includes code from SEA-RAFT (https://github.com/princeton-vl/SEA-RAFT)
# Copyright (c) 2024, Princeton Vision & Learning Lab
# Licensed under the BSD 3-Clause License

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..sea_raft.extractor import ResNetFPN
from ..sea_raft.layer import conv3x3
from ..sea_raft.raft import Recurrent
from ..sea_raft.update import BasicUpdateBlock
from .depth_anything_v2.dpt import DepthAnythingV2

#: Side the depth network sees every frame at.
DEPTH_SIDE = 518

#: Channel statistics the depth input is normalised with.
DEPTH_MEAN = (0.485, 0.456, 0.406)
DEPTH_STD = (0.229, 0.224, 0.225)


class FlowSeek(Recurrent):
    da_model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    }

    def __init__(self, pretrain="resnet18", dim=128, radius=4, num_blocks=2, initial_dim=64, block_dims=(64, 128, 256), da_size="vits"):
        super().__init__()
        self.dim = dim
        self.radius = radius
        self.da_size = da_size
        corr_channel = self.corr_levels * (radius * 2 + 1) ** 2
        backbone = dict(pretrain=pretrain, initial_dim=initial_dim, block_dims=block_dims)
        features = self.da_model_configs[da_size]['features']

        self.cnet = ResNetFPN(input_dim=6, output_dim=2 * dim, **backbone)

        self.dav2 = DepthAnythingV2(**self.da_model_configs[da_size])

        self.merge_head = nn.Sequential(
            nn.Conv2d(features, features//2*3, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features//2*3, features*2, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features*2, features*2, 3, stride=2, padding=1),
        )

        self.bnet = ResNetFPN(input_dim=16, output_dim=2 * dim, **backbone)

        # conv for iter 0 results
        self.init_conv = conv3x3(2 * dim, 2 * dim)

        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(dim*2, dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, 64 * 9, 1, padding=0)
        )
        self.flow_head = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(dim*2, 2 * dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * dim, 6, 3, padding=1)
        )
        self.fnet = ResNetFPN(input_dim=3, output_dim=dim * 2, **backbone)
        self.update_block = BasicUpdateBlock(corr_channel, num_blocks, hdim=dim*2, cdim=dim*2)

    def create_bases(self, disp):
        B, C, H, W = disp.shape
        assert C == 1
        cx = 0.5
        cy = 0.5

        ys = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H)
        xs = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W)
        u, v = torch.meshgrid(xs, ys, indexing='xy')
        u = u - cx
        v = v - cy
        u = u.unsqueeze(0).unsqueeze(0)
        v = v.unsqueeze(0).unsqueeze(0)
        u = u.repeat(B, 1, 1, 1).to(disp.device)
        v = v.repeat(B, 1, 1, 1).to(disp.device)

        aspect_ratio = W / H

        Tx = torch.cat([-torch.ones_like(disp), torch.zeros_like(disp)], dim=1)
        Ty = torch.cat([torch.zeros_like(disp), -torch.ones_like(disp)], dim=1)
        Tz = torch.cat([u, v], dim=1)

        Tx = Tx / torch.linalg.vector_norm(Tx, dim=(1,2,3), keepdim=True)
        Ty = Ty / torch.linalg.vector_norm(Ty, dim=(1,2,3), keepdim=True)
        Tz = Tz / torch.linalg.vector_norm(Tz, dim=(1,2,3), keepdim=True)

        Tx = 2 * disp * Tx
        Ty = 2 * disp * Ty
        Tz = 2 * disp * Tz

        R1x = torch.cat([torch.zeros_like(disp), torch.ones_like(disp)], dim=1)
        R2x = torch.cat([u * v, v * v], dim=1)
        R1y = torch.cat([-torch.ones_like(disp), torch.zeros_like(disp)], dim=1)
        R2y = torch.cat([-u * u, -u * v], dim=1)
        Rz =  torch.cat([-v / aspect_ratio, u * aspect_ratio], dim=1)

        R1x = R1x / torch.linalg.vector_norm(R1x, dim=(1,2,3), keepdim=True)
        R2x = R2x / torch.linalg.vector_norm(R2x, dim=(1,2,3), keepdim=True)
        R1y = R1y / torch.linalg.vector_norm(R1y, dim=(1,2,3), keepdim=True)
        R2y = R2y / torch.linalg.vector_norm(R2y, dim=(1,2,3), keepdim=True)
        Rz =  Rz  / torch.linalg.vector_norm(Rz,  dim=(1,2,3), keepdim=True)

        M = torch.cat([Tx, Ty, Tz, R1x, R2x, R1y, R2y, Rz], dim=1) # Bx(8x2)xHxW
        return M

    def encode(self, images):
        """ Per-frame features, depth and motion bases, images (N, 3, H, W) in [0, 255] """
        N, _, H, W = images.shape
        resized = F.interpolate(images, (DEPTH_SIDE, DEPTH_SIDE), mode="bilinear", align_corners=False) / 255.
        mean = torch.tensor(DEPTH_MEAN, dtype=torch.float64, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(DEPTH_STD, dtype=torch.float64, device=images.device).view(1, 3, 1, 1)
        # The released weights were trained on image / mean - std rather than (image - mean) / std.
        resized = (resized / mean - std).float()

        path1, depth = self.dav2(resized)
        path1 = F.interpolate(path1, (H, W), mode="bilinear", align_corners=False)
        bases = self.create_bases(F.interpolate(depth, (H, W), mode="bilinear", align_corners=False))
        mono = self.merge_head(path1)
        del path1

        image, pad = self.prepare(images)
        bnet = self.init_conv(self.bnet(F.pad(bases, pad, mode='replicate')))
        netbases, ctxbases = torch.split(bnet, [self.dim, self.dim], dim=1)
        fmap = torch.cat((self.fnet(image), mono), 1)
        return {"image": image, "fmap": fmap, "net": netbases, "context": ctxbases, "pad": pad}

    def initial_state(self, first, second):
        net, context = super().initial_state(first, second)
        return torch.cat((net, first["net"]), 1), torch.cat((context, first["context"]), 1)
