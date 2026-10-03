import torch
import torch.nn.functional as F

#: Bytes one chunk of an on-demand lookup may sample at once.
LOCAL_CHUNK_BYTES = 1 << 28


def bilinear_sampler(img, coords):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    return F.grid_sample(img, grid, align_corners=True)

def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(torch.arange(ht, device=device), torch.arange(wd, device=device), indexing="ij")
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

class CorrBlock:
    """ Correlation pyramid between two feature maps, looked up around a flow.

    The all-pairs volume of each level is built once, as upstream does, except for the local
    finest levels: those are correlated on demand at each lookup by sampling the second feature
    pyramid, which gives the same values in memory linear in the frame size.
    """
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4, local=0):
        self.num_levels = num_levels
        self.radius = radius
        self.local = int(local)
        self.dim = fmap1.shape[1]
        self.fmap1 = fmap1 if self.local else None
        r = radius
        dx = torch.linspace(-r, r, 2*r+1, device=fmap1.device)
        dy = torch.linspace(-r, r, 2*r+1, device=fmap1.device)
        self.delta = torch.stack(torch.meshgrid(dy, dx, indexing="ij"), axis=-1).view(1, 2*r+1, 2*r+1, 2)
        self.corr_pyramid = []
        for i in range(self.num_levels):
            if i < self.local:
                self.corr_pyramid.append(fmap2)
            else:
                corr = CorrBlock.corr(fmap1, fmap2)
                batch, h1, w1, h2, w2 = corr.shape
                self.corr_pyramid.append(corr.reshape(batch*h1*w1, 1, h2, w2))
            if i + 1 < self.num_levels:
                fmap2 = F.interpolate(fmap2, scale_factor=0.5, mode='bilinear', align_corners=False)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            coords_lvl = centroid_lvl + self.delta
            if i < self.local:
                corr = self.sampled(i, coords_lvl.view(batch, h1*w1, (2*r+1)**2, 2))
            else:
                corr = bilinear_sampler(self.corr_pyramid[i], coords_lvl)
            out_pyramid.append(corr.view(batch, h1, w1, -1))

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    def sampled(self, level, coords):
        """ Correlation at coords (batch, points, taps, 2) against level of the second pyramid """
        fmap2 = self.corr_pyramid[level]
        batch, dim, H, W = fmap2.shape
        fmap1 = self.fmap1.flatten(2)
        xgrid, ygrid = coords.split([1,1], dim=-1)
        grid = torch.cat([2*xgrid/(W-1) - 1, 2*ygrid/(H-1) - 1], dim=-1)
        points, taps = grid.shape[1:3]
        step = max(1, LOCAL_CHUNK_BYTES // (batch * dim * taps * fmap2.element_size()))
        out = []
        for start in range(0, points, step):
            taken = F.grid_sample(fmap2, grid[:, start:start + step], align_corners=True)
            out.append((taken * fmap1[:, :, start:start + step, None]).sum(1))
        return torch.cat(out, dim=1) / torch.sqrt(torch.tensor(self.dim).float())

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, h1, w1 = fmap1.shape
        h2, w2 = fmap2.shape[2:]
        fmap1 = fmap1.view(batch, dim, h1*w1)
        fmap2 = fmap2.view(batch, dim, h2*w2)
        corr = fmap1.transpose(1, 2) @ fmap2
        corr = corr.reshape(batch, h1, w1, h2, w2)
        return corr.div_(torch.sqrt(torch.tensor(dim).float()))
