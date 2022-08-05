import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time
from lib.utils import get_sobel_kernel


class SingleScaleConvEncoder(nn.Module):
    def __init__(self, resolution, latent_dim=4, output_dim=2, upscale_res=None, channel_dim=32, kernel_size=3):
        super(SingleScaleConvEncoder, self).__init__()
        assert kernel_size % 2 == 1, "kernel size must be odd"
        self.resolution = resolution
        self.world_size = torch.tensor([resolution] * 3).long()
        self.grid = torch.nn.Parameter(torch.zeros(1, latent_dim, *self.world_size).uniform_(-1e-4, 1e-4))
        self.upscale_res = upscale_res
        if upscale_res is not None and resolution != upscale_res:
            self.upscale_res = torch.tensor([upscale_res] * 3).long()

        padding_size = int((kernel_size - 1) / 2)
        self.conv_1 = nn.Sequential(
            nn.Conv3d(latent_dim, channel_dim, kernel_size=kernel_size, stride=1, padding=padding_size),
            nn.BatchNorm3d(channel_dim),
            nn.PReLU(channel_dim)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv3d(channel_dim + latent_dim, channel_dim, kernel_size=kernel_size, stride=1, padding=padding_size),
            nn.BatchNorm3d(channel_dim),
            nn.PReLU(channel_dim)
        )
        if upscale_res is None:
            self.conv_3 = nn.ConvTranspose3d(channel_dim * 2 + latent_dim, output_dim, kernel_size=kernel_size,
                                             padding=padding_size, stride=2, output_padding=1),
        else:
            self.conv_3 = nn.Sequential(
                nn.ConvTranspose3d(channel_dim * 2 + latent_dim, output_dim, kernel_size=kernel_size,
                                   padding=padding_size, stride=2, output_padding=1),
                nn.BatchNorm3d(output_dim),
                nn.PReLU(output_dim)
            )

    def forward(self):
        x1 = self.conv_1(self.grid)
        x = torch.cat([self.grid, x1], dim=1)
        x2 = self.conv_2(x)
        x = torch.cat([x, x2], dim=1)
        x3 = self.conv_3(x)

        if self.upscale_res is not None:
            x3 = F.interpolate(x3, size=tuple(self.upscale_res), mode='trilinear', align_corners=True)

        return x3


class MultiScaleConvEncoder(nn.Module):
    def __init__(self, num_levels=4, level_dim=4, base_resolution=16, desired_resolution=128,
                 per_level_scale=2, output_dim=1):
        super(MultiScaleConvEncoder, self).__init__()
        self.base_resolution = base_resolution // 2
        self.desired_resolution = desired_resolution // 2
        if desired_resolution is not None:
            per_level_scale = np.exp2(np.log2(self.desired_resolution / self.base_resolution) / (num_levels - 1))
        self.num_levels = num_levels
        self.level_dim = level_dim
        self.per_level_scale = per_level_scale
        self.encoders = nn.ModuleList()
        self.output_dim = output_dim

        for i in range(num_levels):
            resolution = int(np.ceil(self.base_resolution * self.per_level_scale ** i))
            self.encoders.append(SingleScaleConvEncoder(resolution=resolution, output_dim=level_dim,
                                                        upscale_res=desired_resolution))

        self.output_layer = nn.Conv3d(in_channels=num_levels * level_dim, out_channels=self.output_dim, kernel_size=1,
                                      padding=0, stride=1)

    def forward(self):
        level_outs = []
        for encoder in self.encoders:
            level_outs.append(encoder.forward())

        net_out = torch.cat(level_outs, dim=1)

        return self.output_layer(net_out)


class GridReg(nn.Module):
    def __init__(self, rgb_latent_dim):
        super(GridReg, self).__init__()
        self.encoder = MultiScaleConvEncoder(output_dim=1 + rgb_latent_dim)
        self.sobel_kernel = get_sobel_kernel(kernel_size=3).cuda()

    def forward(self, calc_normals=False):
        x = self.encoder.forward()
        if calc_normals:
            density = x[:, :1, ...]
            normals = -F.normalize(F.conv3d(density, self.sobel_kernel, stride=1, padding=2), dim=1)
        else:
            normals = None

        # wadu = F.interpolate(x[:, :1, ...], size=(256, 256, 256), mode='trilinear').squeeze().cpu().detach().numpy()
        # import mcubes, trimesh
        # vertices, triangles = mcubes.marching_cubes(wadu, 0.1)
        # mesh = trimesh.Trimesh(vertices, triangles)
        # mesh.export("target.obj")
        return x, normals


if __name__ == "__main__":
    encoder = MultiScaleConvEncoder().cuda()
    for i in range(10):
        start_time = time.time()
        output = encoder.forward()
        print("forward pass in: ", time.time() - start_time)
        del output