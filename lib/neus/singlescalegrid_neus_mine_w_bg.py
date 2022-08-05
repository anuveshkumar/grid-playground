import mcubes
import torch
import torch.nn as nn
from lib.embedder import get_embedder
import torch.nn.functional as F
import numpy as np
import cv2
from lib.bg_model import NeRF, get_sphere_intersection
from lib import bg_model
from lib.utils import *


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
        self.radius = self.xyz_max[0]
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
        self.background_nlayers = 32
        self.background_reso = 256
        self.model_background = kwargs['model_background']
        if self.model_background is True:
            self.background_data = nn.Parameter(
                torch.zeros([
                    1, 4,
                    self.background_reso * 2,
                    self.background_reso,
                    self.background_nlayers,
                ])
            )

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, **render_kwargs):
        ray_pts, mask_outbox, z_vals = self.sample_ray(rays_o, rays_d, is_train=global_step is not None,
                                                       **render_kwargs)

        if self.model_background:
            m = ray_pts.norm(dim=-1) <= self.radius
            mask_outbox = mask_outbox | ~m

        batch_size, n_samples, _ = ray_pts.shape
        sdfs = torch.ones((batch_size, n_samples)) * 100
        rgbs = torch.zeros_like(ray_pts)
        normals = torch.zeros_like(ray_pts)
        sdf_out, feature_out, gradients_out = self.implicit_network(ray_pts[~mask_outbox])
        sdfs[~mask_outbox] = sdf_out

        viewdirs_exp = viewdirs.unsqueeze(1).repeat(1, n_samples, 1)
        normals[~mask_outbox] = F.normalize(gradients_out, dim=-1)
        refdirs = (2 * (torch.matmul(-viewdirs_exp.unsqueeze(-2), normals.unsqueeze(-1)).squeeze(-2))
                   * normals + viewdirs_exp)[~mask_outbox]
        angle_v_n = torch.matmul(-viewdirs_exp.unsqueeze(-2), normals.unsqueeze(-1)).squeeze(-2)[~mask_outbox]
        angle_n_d = torch.matmul(normals.unsqueeze(-2), viewdirs_exp.unsqueeze(-1)).squeeze(-2)
        refdirs_emb = self.directional_encoder(refdirs)
        # viewdirs_emb = self.directional_encoder(viewdirs_exp)
        rgb_in = torch.cat([feature_out, angle_v_n, refdirs_emb, ray_pts[~mask_outbox]], dim=-1)
        rgb_out = self.directional_mlp(rgb_in)

        rgbs[~mask_outbox] = rgb_out

        weights, bg_transmittance = self.volume_rendering(sdfs)

        rgb = torch.sum(weights[..., None] * rgbs[:, :-1, :], 1)
        distances = (rays_o[..., None, :] - ray_pts[..., :-1, :]).norm(dim=-1)
        depth = (weights * distances).sum(dim=-1)

        if self.model_background:
            bg_rgb, bg_depth = self.model_background_w_msi(rays_o, rays_d, viewdirs, render_kwargs)
            rgb += bg_transmittance[..., None] * bg_rgb
            depth += bg_transmittance * bg_depth

        render_result = {
            'rgb': rgb,
            'depth': depth,
            'transmittance': bg_transmittance,
            'angle_n_d': angle_n_d[:, -1, :],
            'weights': weights
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
        N_samples = int(np.linalg.norm(np.array(self.implicit_network.world_size.cpu().numpy() * 2) + 1) / stepsize) + 1

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

    def model_background_w_msi(self, rays_o, rays_d, viewdirs, render_kwargs):
        world_size = self.implicit_network.world_size
        delta_scale = 1 / (viewdirs * (0.5 * world_size).to(device=rays_o.device)).norm(dim=-1).reshape(-1)
        ray_shape = None
        if len(rays_o.shape) > 2:
            ray_shape = rays_o.shape
            rays_o = rays_o.reshape(-1, 3)
            viewdirs = viewdirs.reshape(-1, 3)
        with torch.no_grad():
            origins = bg_model.world2grid(rays_o, world_size)
            csi = bg_model.ConcentricSpheresIntersector(
                world_size,
                origins,
                viewdirs,
                delta_scale)
            inner_radius = torch.cross(csi.origins, csi.dirs, dim=-1).norm(dim=-1) + 1e-3
            inner_radius = inner_radius.clamp_min(1.0)
            _, t_last = csi.intersect(inner_radius)
            n_steps = int(self.background_nlayers / render_kwargs['stepsize']) + 2
            # log_light_intensity = alphainv_cum[...,-1].reshape(-1).clone()

            ## parallelized testing code.
            r = n_steps / (n_steps - 0.5 - torch.arange(n_steps).to(rays_o.device)).unsqueeze(0)
            active_mask_all, t_all = csi.intersect(r)
            active_mask_all = active_mask_all & (r >= inner_radius.unsqueeze(-1))

            t_mid_sub_all = (t_all + t_last.unsqueeze(-1).expand(-1, t_all.shape[1])) * 0.5
            sphpos_all = csi.origins.unsqueeze(1) + t_mid_sub_all.unsqueeze(-1) * csi.dirs.unsqueeze(1)
            invr_mid_all = 1.0 / torch.norm(sphpos_all, dim=-1)
            sphpos_all = sphpos_all * invr_mid_all.unsqueeze(-1)

            xy = bg_model.xyz2equirect(sphpos_all, self.background_reso)
            z = torch.clamp((1.0 - invr_mid_all) * self.background_nlayers - 0.5, 0.0,
                            self.background_nlayers - 1)
            points = torch.cat([xy, z.unsqueeze(-1)], dim=-1)

            bg_shape = torch.tensor(self.background_data.shape[2:]).to(rays_o.device)
            points = (points / (bg_shape[None, None, :] / 2)) - 1
            t_all = torch.cat([t_last.unsqueeze(-1), t_all], dim=-1)  # test
            interval_bg = t_all[..., 1:] - t_all[..., :-1]  # t_all-t_last[...,None]
            interval_bg = csi.world_step_scale[..., None].expand(*interval_bg.shape) * interval_bg * 10

        rgba_bg = torch.zeros(*points.shape[:2], 4).to(points)
        alpha_bg = torch.zeros(*points.shape[:2]).to(points)

        # rgba[active_mask_all] = self.grid_sampler(points[active_mask_all], self.background_data.reshape(1,512,256,32,4).permute(0,4,1,2,3)) # used for debugging only
        rgba_bg[active_mask_all] = self.grid_sampler(points[active_mask_all], self.background_data)
        alpha_bg[active_mask_all] = self.activate_density(rgba_bg[active_mask_all][..., -1],
                                                          interval_bg[active_mask_all])
        # alpha_bg[active_mask_all] = F.relu(rgba_bg[active_mask_all][...,-1])
        rgba_bg[..., :3] = torch.clamp_min(rgba_bg[..., :3] * bg_model.SH_C0 + 0.5, 0.0)
        if ray_shape is not None:
            rays_o = rays_o.reshape(ray_shape)
            viewdirs = viewdirs.reshape(ray_shape)
            rgba_bg = rgba_bg.reshape(*ray_shape[:-1], *rgba_bg.shape[-2:])
            alpha_bg = alpha_bg.reshape(*ray_shape[:-1], *alpha_bg.shape[-1:])
            points = points.reshape(*ray_shape[:-1], *points.shape[-2:])
            sphpos_all = sphpos_all.reshape(*ray_shape[:-1], *sphpos_all.shape[-2:])

        weights, _ = get_ray_marching_ray(alpha_bg)
        rgb_marched = (weights[..., None] * rgba_bg[..., :3]).sum(dim=-2)
        depth_marched = (weights * t_mid_sub_all).sum(dim=-1)
        return rgb_marched, depth_marched



def cumprod_exclusive(p):
    # Not sure why: it will be slow at the end of training if clamping at 1e-10 is not applied
    return torch.cat([torch.ones_like(p[..., [0]]), p.clamp_min(1e-10).cumprod(-1)], -1)


def get_ray_marching_ray(alpha):
    alphainv_cum = cumprod_exclusive(1 - alpha)
    weights = alpha * alphainv_cum[..., :-1]
    return weights, alphainv_cum
