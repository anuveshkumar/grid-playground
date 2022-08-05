import mcubes
import torch
import torch.nn as nn
from lib.embedder import get_embedder
import torch.nn.functional as F
import numpy as np
import cv2


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self):
        return torch.exp(self.variance * 10.0)


class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''

    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height,
                                             in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)


class ImplicitNetwork(nn.Module):
    def __init__(self, xyz_min, xyz_max, resolution, latent_dim, feature_size):
        super(ImplicitNetwork, self).__init__()
        setattr(self, 'xyz_min', xyz_min)
        setattr(self, 'xyz_max', xyz_max)
        self._set_grid_resolution(resolution)

        self.latent_dim = latent_dim
        self.grid = torch.nn.Parameter(torch.zeros([1, self.latent_dim, *self.world_size]))
        self.conv1 = nn.Sequential(
            nn.Conv3d(self.latent_dim, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(64 + self.latent_dim, 64, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        # self.conv3 = nn.Conv3d(64, (1 + feature_size) * 8, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv3 = nn.ConvTranspose3d(64 * 2 + self.latent_dim, 1 + feature_size, kernel_size=3, padding=1, stride=2,
                                        bias=True,
                                        output_padding=1)
        # self.pixel_shuffle = PixelShuffle3d(2)
        self._get_sobel_kernel(kernel_size=5)
        self.finetune = False

    def _forward(self, sdf_only=False):
        if not self.finetune:
            x1 = self.conv1(self.grid)
            x = torch.cat([self.grid, x1], dim=1)
            x2 = self.conv2(x)
            x = torch.cat([x, x2], dim=1)
            x = self.conv3(x)
        else:
            x = self.grid
        # x = self.pixel_shuffle(x)
        density = x[:, :1, ...]
        if sdf_only:
            return density

        feature = x[:, 1:, ...]
        gradients = F.conv3d(density, self.sobel_kernel, stride=1, padding=2)

        return density, feature, gradients

    def forward(self, xyz):
        sdf_grid, feature_grid, gradient_grid = self._forward()
        sdf, features, gradients = self.grid_sampler(xyz, sdf_grid, feature_grid, gradient_grid)
        return sdf, features, gradients

    def get_sdf_vals(self, xyz):
        sdf_grid = self._forward(sdf_only=True)
        sdf = self.grid_sampler(xyz, sdf_grid)
        return sdf

    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True):
        '''Wrapper for the interp operation'''
        mode = 'bilinear'
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        ret_lst = [
            F.grid_sample(grid, ind_norm, mode=mode, align_corners=align_corners)
            .reshape(grid.shape[1], -1).T.reshape(*shape, grid.shape[1]) for grid in grids
        ]
        for i in range(len(grids)):
            if ret_lst[i].shape[-1] == 1:
                ret_lst[i] = ret_lst[i].squeeze(-1)
        if len(ret_lst) == 1:
            return ret_lst[0]
        return ret_lst

    def _get_sobel_kernel(self, kernel_size):
        h_dash, h = cv2.getDerivKernels(1, 0, kernel_size, normalize=True)
        h_dash = h_dash
        h = h
        kernel_x = np.outer(np.outer(h_dash, h), h).reshape(kernel_size, kernel_size, kernel_size)
        kernel_y = np.outer(np.outer(h, h_dash), h).reshape(kernel_size, kernel_size, kernel_size)
        kernel_z = np.outer(np.outer(h, h), h_dash).reshape(kernel_size, kernel_size, kernel_size)

        self.sobel_kernel = torch.from_numpy(
            np.array([kernel_x, kernel_y, kernel_z]).reshape(
                (3, 1, kernel_size, kernel_size, kernel_size))).float().cuda()
        # self.sobel_kernel /= self.sobel_kernel.abs().max()

    @torch.no_grad()
    def scale_volume_grid(self, resolution):
        ori_world_size = self.world_size
        self._set_grid_resolution(resolution)
        print('scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)
        self.grid = torch.nn.Parameter(
            F.interpolate(self.grid.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))

        print('scale_volume_grid finished')

    def _set_grid_resolution(self, resolution):
        self.resolution = resolution
        self.world_size = torch.tensor([resolution] * 3).long()
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / (resolution ** 3)).pow(1 / 3)

    @torch.no_grad()
    def _finetune(self):
        sdfs, features, gradients = self._forward()
        self.grid = torch.nn.Parameter(torch.cat([sdfs, features], dim=1))
        self.finetune = True


class SingleScaleNeUS(nn.Module):
    def __init__(self, xyz_min, xyz_max, resolution, latent_dim=12, rgb_latent_dim=15, **kwargs):
        super(SingleScaleNeUS, self).__init__()
        setattr(self, 'xyz_min', xyz_min)
        setattr(self, 'xyz_max', xyz_max)

        self.latent_dim = latent_dim
        self.rgb_latent_dim = rgb_latent_dim

        self.deviation_network = SingleVarianceNetwork(init_val=0.1)

        self.implicit_network = ImplicitNetwork(xyz_min=xyz_min, xyz_max=xyz_max, resolution=resolution,
                                                latent_dim=latent_dim, feature_size=rgb_latent_dim)

        self.directional_encoder, output_dim = get_embedder(multires=6, input_dims=3, include_inputs=True)
        dim0 = self.rgb_latent_dim + output_dim + 4

        self.directional_mlp = nn.Sequential(
            nn.Linear(dim0, 64), nn.ReLU(inplace=True),
            *[nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
              for _ in range(2)
              ],
            nn.Linear(64, 3)
        )

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, **render_kwargs):
        ray_pts, mask_outbox, z_vals = self.sample_ray(rays_o, rays_d, is_train=global_step is not None,
                                                       **render_kwargs)
        batch_size, n_samples, _ = ray_pts.shape
        sdfs = torch.ones((batch_size, n_samples)) * 100
        rgbs = torch.zeros_like(ray_pts)

        sdf_out, feature_out, gradients_out = self.implicit_network(ray_pts[~mask_outbox])

        sdfs[~mask_outbox] = sdf_out

        viewdirs = viewdirs.unsqueeze(1).repeat(1, n_samples, 1)[~mask_outbox]
        normals = F.normalize(gradients_out, dim=-1)
        refdirs = (2 * (torch.matmul(-viewdirs.unsqueeze(-2), normals.unsqueeze(-1)).squeeze(-2)) * normals + viewdirs)
        angle_v_n = torch.matmul(-viewdirs.unsqueeze(-2), normals.unsqueeze(-1)).squeeze(-2)
        refdirs_emb = self.directional_encoder(refdirs)
        # viewdirs_emb = self.directional_encoder(viewdirs)
        rgb_in = torch.cat([feature_out, angle_v_n, refdirs_emb, ray_pts[~mask_outbox]], dim=-1)
        rgb_out = self.directional_mlp(rgb_in)

        rgbs[~mask_outbox] = rgb_out

        weights, bg_transmittance = self.volume_rendering(sdfs)

        rgb = torch.sum(weights[..., None] * rgbs[:, :-1, :], 1)
        distances = (rays_o[..., None, :] - ray_pts[..., :-1, :]).norm(dim=-1)
        depth = (weights * distances).sum(dim=-1)

        render_result = {
            'rgb': rgb,
            'depth': depth,
        }
        return render_result

    def volume_rendering(self, sdf):
        scale = self.deviation_network()
        s_density = self.sample_logistic_cumulative_density_distribution(sdf, scale)
        alpha = torch.clip((s_density[..., :-1] - s_density[..., 1:]) / (s_density[..., :-1] + 1e-10), 0)  # prevent nan
        acc_transmittance = torch.cumprod((1 - alpha), -1)
        bg_lambda = acc_transmittance[..., -1]
        acc_transmittance = torch.roll(acc_transmittance, 1, -1)
        acc_transmittance[..., :1] = 1
        weights = acc_transmittance * alpha

        return weights, bg_lambda

    def sample_logistic_cumulative_density_distribution(self, location, scale):
        return 1 / (1 + torch.exp(-location * scale))

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, is_train=False, **render_kwargs):
        # 1. max number of query points to cover all possible rays
        N_samples = int(np.linalg.norm(np.array(self.implicit_network.world_size.cpu().numpy()) + 1) / stepsize) + 1

        # determine the two end-points of ray bbox intersection
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.xyz_max - rays_o) / vec
        rate_b = (self.xyz_min - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
        t_max = torch.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
        # 3. check wheter a raw intersect the bbox or not
        mask_outbbox = (t_max <= t_min)
        # 4. sample points on each ray
        rng = torch.arange(N_samples)[None].float().to(rays_o.device)
        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1)
            rng += torch.rand_like(rng[:, [0]])
        step = stepsize * self.implicit_network.voxel_size * rng
        interpx = (t_min[..., None] + step / rays_d.norm(dim=-1, keepdim=True))
        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        # 5. update mask for query points outside bbox
        mask_outbbox = mask_outbbox[..., None] | ((self.xyz_min > rays_pts) | (rays_pts > self.xyz_max)).any(dim=-1)
        return rays_pts, mask_outbbox, interpx

    def extract_geometry(self, resolution, threshold):
        query_func = lambda pts: -self.implicit_network.get_sdf_vals(pts)
        u = self.extract_fields(resolution, query_func)
        vertices, triangles = mcubes.marching_cubes(u, threshold)

        b_max_np = self.xyz_max.detach().cpu().numpy()
        b_min_np = self.xyz_min.detach().cpu().numpy()

        vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
        return vertices, triangles

    def extract_fields(self, resolution, query_func):
        N = 64
        X = torch.linspace(self.xyz_min[0], self.xyz_max[0], resolution).split(N)
        Y = torch.linspace(self.xyz_min[1], self.xyz_max[1], resolution).split(N)
        Z = torch.linspace(self.xyz_min[2], self.xyz_max[2], resolution).split(N)

        u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs)
                        pts = torch.cat([
                            xx.reshape(-1, 1),
                            yy.reshape(-1, 1),
                            zz.reshape(-1, 1)
                        ],
                            dim=-1)
                        val = query_func(pts).reshape(
                            len(xs), len(ys), len(zs)).detach().cpu().numpy()
                        u[xi * N:xi * N + len(xs), yi * N:yi * N + len(ys),
                        zi * N:zi * N + len(zs)] = val
        return u

    def get_kwargs(self):
        return {
            "xyz_min": self.xyz_min,
            "xyz_max": self.xyz_max,
            "resolution": self.implicit_network.resolution,
        }
