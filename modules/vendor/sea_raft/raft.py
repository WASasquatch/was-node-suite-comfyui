import torch
import torch.nn as nn
import torch.nn.functional as F

from .corr import CorrBlock, coords_grid
from .extractor import ResNetFPN
from .layer import conv3x3
from .update import BasicUpdateBlock


def padding(ht, wd):
    """ Replicate padding that makes both sides divisible by 8, as [left, right, top, bottom] """
    pad_ht = (((ht // 8) + 1) * 8 - ht) % 8
    pad_wd = (((wd // 8) + 1) * 8 - wd) % 8
    return [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]

def unpad(x, pad):
    ht, wd = x.shape[-2:]
    c = [pad[2], ht-pad[3], pad[0], wd-pad[1]]
    return x[..., c[0]:c[1], c[2]:c[3]]


class Recurrent(nn.Module):
    """ The recurrent refinement SEA-RAFT runs, shared with the networks built on it.

    encode() runs once per frame; estimate() runs once per ordered pair of encoded frames.
    """
    corr_levels = 4

    def prepare(self, images):
        """ images: (N, 3, H, W) in [0, 255] -> normalised and padded, and the padding used """
        images = 2 * (images / 255.0) - 1.0
        pad = padding(*images.shape[-2:])
        return F.pad(images.contiguous(), pad, mode='replicate'), pad

    def upsample_flow(self, flow, mask):
        """ Upsample [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def initial_state(self, first, second):
        cnet = self.cnet(torch.cat([first["image"], second["image"]], dim=1))
        cnet = self.init_conv(cnet)
        net, context = torch.split(cnet, [self.dim, self.dim], dim=1)
        return net, context

    def estimate(self, first, second, iters, local=0):
        """ Flow from each frame of first onto the same entry of second, both as encode() gave them.

        local is how many of the finest correlation levels are computed on demand.
        """
        net, context = self.initial_state(first, second)

        # init flow
        flow_8x = self.flow_head(net)[:, :2]

        if iters > 0:
            corr_fn = CorrBlock(first["fmap"], second["fmap"], self.corr_levels, self.radius, local=local)
            N, _, H, W = flow_8x.shape
            grid = coords_grid(N, H, W, device=flow_8x.device)

        for itr in range(iters):
            corr = corr_fn(grid + flow_8x)
            net = self.update_block(net, context, corr, flow_8x)
            flow_8x = flow_8x + self.flow_head(net)[:, :2]

        flow_up = self.upsample_flow(flow_8x, .25 * self.upsample_weight(net))
        return unpad(flow_up, first["pad"])

    def forward(self, image1, image2, iters=4):
        """ Estimate optical flow between pair of frames, both (N, 3, H, W) in [0, 255] """
        return self.estimate(self.encode(image1), self.encode(image2), iters)


class RAFT(Recurrent):
    def __init__(self, pretrain="resnet34", dim=128, radius=4, num_blocks=2, initial_dim=64, block_dims=(64, 128, 256)):
        super().__init__()
        self.dim = dim
        self.radius = radius
        corr_channel = self.corr_levels * (radius * 2 + 1) ** 2
        backbone = dict(pretrain=pretrain, initial_dim=initial_dim, block_dims=block_dims)
        self.cnet = ResNetFPN(input_dim=6, output_dim=2 * dim, **backbone)

        # conv for iter 0 results
        self.init_conv = conv3x3(2 * dim, 2 * dim)
        self.upsample_weight = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(dim, dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, 64 * 9, 1, padding=0)
        )
        self.flow_head = nn.Sequential(
            # flow(2) + weight(2) + log_b(2)
            nn.Conv2d(dim, 2 * dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * dim, 6, 3, padding=1)
        )
        self.fnet = ResNetFPN(input_dim=3, output_dim=dim * 2, **backbone)
        self.update_block = BasicUpdateBlock(corr_channel, num_blocks, hdim=dim, cdim=dim)

    def encode(self, images):
        """ Per-frame features, images (N, 3, H, W) in [0, 255] """
        image, pad = self.prepare(images)
        return {"image": image, "fmap": self.fnet(image), "pad": pad}
