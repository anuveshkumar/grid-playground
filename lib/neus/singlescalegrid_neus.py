import mcubes
import numpy as np
import torch
import torch.nn as nn

from lib.embedder import get_embedder
import torch.nn.functional as F
from lib.utils import grid_sample
import cv2


class SingleScaleNeus(nn.Module):
    def __init__(self,
                 xyz_min=[-1.0] * 3,
                 xyz_max=[1.0] * 3,
                 resolution=64,
                 latent_dim=12,
                 rgb_latent_dim=15,
                 **kwargs):
        super(SingleScaleNeus, self).__init__()

        self.register_buffer('xyz_min', torch.tensor(xyz_min))
        self.register_buffer('xyz_max', torch.tensor(xyz_max))
        self.latent_dim = latent_dim
        self.rgb_latent_dim = rgb_latent_dim
        self._set_grid_resolution(resolution)
        self.deviation_network = SingleVarianceNetwork(init_val=0.3)

        self.grid = torch.nn.Parameter(torch.zeros([1, self.latent_dim, *self.world_size]))
        self.grid_reg = GridReg(latent_dim=self.latent_dim, rgb_latent_dim=self.rgb_latent_dim)

        self.directional_encoder, output_dim = get_embedder(multires=6, input_dims=3, include_inputs=True)
        dim0 = self.rgb_latent_dim + output_dim + 3  # feature size, viewdirs_emb, pts, normals, angle_v_n

        self.directional_mlp = nn.Sequential(
            nn.Linear(dim0, 64), nn.ReLU(inplace=False),
            *[nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=False))
              for _ in range(2)
              ],
            nn.Linear(64, 3)
        )

        self.saved_grids = None

    def initialize_grids(self):
        self.saved_grids = self.grid_reg.forward(self.grid, calc_normals=True)

    def _set_grid_resolution(self, resolution):
        self.resolution = resolution
        self.world_size = torch.tensor([resolution] * 3).long()
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / (resolution ** 3)).pow(1 / 3)

    def extract_latent(self, xyz):
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip(-1) * 2 - 1
        if self.saved_grids is None:
            grid_reg, gradients = self.grid_reg.forward(self.grid, calc_normals=True)
        else:
            grid_reg, gradients = self.saved_grids
        # grid_reg = self.grid
        latents = F.grid_sample(grid_reg, ind_norm, align_corners=True).reshape(grid_reg.shape[1],
                                                                                -1).T.reshape(*shape, grid_reg.shape[
            1]).squeeze()
        if gradients is not None:
            gradients = F.grid_sample(gradients, ind_norm, align_corners=True).reshape(gradients.shape[1],
                                                                                       -1).T.reshape(*shape,
                                                                                                     gradients.shape[
                                                                                                         1]).squeeze()
            # normals = F.normalize(normals, dim=-1)
        else:
            gradients = torch.zeros_like(xyz)

        return latents, gradients

    def get_sdf_output(self, pts, mask_outbox=None):
        batch_size, n_samples, _ = pts.shape
        sdf_out = torch.cat(
            (torch.ones(batch_size, n_samples, 1) * 100, torch.zeros(batch_size, n_samples, self.rgb_latent_dim)),
            dim=-1)
        gradients_out = torch.zeros_like(pts)
        latents, grads = self.extract_latent(pts[~mask_outbox])
        sdf_out[~mask_outbox] = latents
        gradients_out[~mask_outbox] = grads
        return sdf_out, gradients_out

    def get_sdf_output_test(self, pts):
        latents, normals = self.extract_latent(pts)
        return latents[..., :1]

    def get_gradient(self, pts, mask_outbox):
        pts.requires_grad_(True)
        sdf_out, _ = self.get_sdf_output(pts, mask_outbox)
        y = sdf_out[..., :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=pts,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def forward(self, rays_o, rays_d, viewdirs, cos_anneal_ratio=0.0, global_step=None, **render_kwargs):

        ray_pts, mask_outbox, z_vals = self.sample_ray(rays_o, rays_d, is_train=global_step is not None,
                                                       **render_kwargs)

        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, dists[..., -2:-1]], dim=-1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]
        dirs = rays_d[:, None, :].expand(pts.shape)
        viewdirs = viewdirs[:, None, :].expand(pts.shape)

        N, Ns, _ = pts.shape

        # pts = pts.reshape(-1, 3)
        # viewdirs = viewdirs.reshape(-1, 3)
        sdf_output, gradients_sobel = self.get_sdf_output(pts, mask_outbox)
        sdf = sdf_output[..., :1]
        feature_vector = sdf_output[..., 1:]

        # gradients_deriv = self.get_gradient(pts, mask_outbox)

        # viewdirs_emb = self.directional_encoder(viewdirs)
        normals = -F.normalize(gradients_sobel, dim=-1)
        # refdirs = (2 * (torch.matmul(-viewdirs.unsqueeze(-2), normals.unsqueeze(-1)).squeeze(-2)) * normals + viewdirs)
        # angle_v_n = torch.matmul(-viewdirs.unsqueeze(-2), normals.unsqueeze(-1)).squeeze(-2)
        # refdirs_emb = self.directional_encoder(refdirs)
        #
        if global_step < 1000:
            dirs_emb = self.directional_encoder(viewdirs)
        else:
            refdirs = (2 * (
                torch.matmul(-viewdirs.unsqueeze(-2), normals.unsqueeze(-1)).squeeze(-2)) * normals + viewdirs)
            # angle_v_n = torch.matmul(-viewdirs.unsqueeze(-2), normals.unsqueeze(-1)).squeeze(-2)
            dirs_emb = self.directional_encoder(refdirs)
        dir_input = torch.cat((pts, dirs_emb, feature_vector), dim=-1)
        sampled_color = torch.zeros_like(pts)
        sampled_color[~mask_outbox] = torch.sigmoid(self.directional_mlp(dir_input[~mask_outbox]))

        inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)  # Single parameter
        inv_s = inv_s.expand(batch_size, n_samples, 1)
        true_cos = (dirs * gradients_sobel).sum(-1, keepdim=True)

        iter_cos = -F.relu(-true_cos * 0.5 + 0.5)

        estimated_next_sdf = sdf + iter_cos * dists[..., None] * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists[..., None] * 0.5
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
        p = prev_cdf - next_cdf
        c = prev_cdf
        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        transmittance = torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)
        weights = alpha * transmittance[:, :-1]

        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        acc = weights.sum(dim=-1, keepdim=True)
        depth = (weights * mid_z_vals).sum(dim=-1)
        normals_marched = (normals * weights[:, :, None]).sum(dim=1)
        disp = 1 / depth

        # bg_rgb = render_kwargs['bg']
        # bg_depth = render_kwargs['far']

        ret_dict = {
            'weights': weights,
            'rgb': color,
            'raw_rgb': sampled_color,
            'depth': depth,
            'acc': acc,
            'grads': normals_marched,
            'disp': disp,
            # 'gradients_deriv': gradients_deriv,
            'gradients_sobel': gradients_sobel,
        }

        for key in ret_dict.keys():
            if torch.isnan(ret_dict[key]).sum() > 0:
                print("checkpoint")

        return ret_dict

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, is_train=False, **render_kwargs):
        # 1. max number of query points to cover all possible rays
        # world_size = render_kwargs['world_size']
        N_samples = int(np.linalg.norm(np.array(self.world_size.cpu().numpy() * 2) + 1) / stepsize) + 1

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
        step = stepsize * self.voxel_size * rng
        interpx = (t_min[..., None] + step / rays_d.norm(dim=-1, keepdim=True))
        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        # 5. update mask for query points outside bbox
        mask_outbbox = mask_outbbox[..., None] | ((self.xyz_min > rays_pts) | (rays_pts > self.xyz_max)).any(dim=-1)
        return rays_pts, mask_outbbox, interpx

    def get_kwargs(self):
        return {
            "xyz_min": self.xyz_min,
            "xyz_max": self.xyz_max,
            "resolution": self.resolution,
        }

    @torch.no_grad()
    def scale_volume_grid(self, resolution):
        ori_world_size = self.world_size
        self._set_grid_resolution(resolution)
        print('scale_volume_grid scale world_size from', ori_world_size, 'to', self.world_size)
        self.grid = torch.nn.Parameter(
            F.interpolate(self.grid.data, size=tuple(self.world_size), mode='trilinear', align_corners=True))

        # self.offset_grid = torch.nn.Parameter(torch.zeros([1, 16, *(self.world_size * 2)]))
        print('scale_volume_grid finished')

    def extract_geometry(self, resolution, threshold):
        print(f'threshold: {threshold}')
        query_func = lambda pts: -self.get_sdf_output_test(pts)
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




class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)


class GridReg(nn.Module):
    def __init__(self, latent_dim, rgb_latent_dim, n_channels=32):
        super(GridReg, self).__init__()
        # network
        self.conv_1 = nn.Conv3d(latent_dim, n_channels, kernel_size=3, padding=1, bias=True)
        self.conv_2 = nn.Conv3d(n_channels, n_channels, kernel_size=3, padding=1, bias=True)
        self.conv_3 = nn.ConvTranspose3d(n_channels, 1 + rgb_latent_dim, kernel_size=3, stride=2, padding=1,
                                         output_padding=1, bias=True)
        # self.conv_3 = nn.Conv3d(64, 1 + rgb_latent_dim, kernel_size=1, stride=1, bias=True)
        self.act_fn_1 = nn.PReLU(num_parameters=n_channels)
        self.act_fn_2 = nn.PReLU(num_parameters=n_channels)

        self.conv_1_bn = nn.BatchNorm3d(n_channels)
        self.conv_2_bn = nn.BatchNorm3d(n_channels)
        # self.conv_3_bn = nn.BatchNorm3d()
        self._get_sobel_kernel(kernel_size=5)

    def forward(self, grid, calc_normals=False):
        x = self.act_fn_1(self.conv_1_bn(self.conv_1(grid)))
        x = self.act_fn_2(self.conv_2_bn(self.conv_2(x)))
        x = self.conv_3(x)
        # x += F.interpolate(grid, size=x.shape[2:], mode='trilinear')
        if calc_normals:
            density = x[:, :1, ...]
            gradients = F.conv3d(density, self.sobel_kernel, stride=1, padding=2)
        else:
            gradients = None
        # normals = -F.normalize(self.sobel_kernel(x[:, :1, ...]))
        return x, gradients

    def _get_sobel_kernel(self, kernel_size=5):
        h_dash, h = cv2.getDerivKernels(1, 0, kernel_size)
        h_dash = h_dash
        h = h
        kernel_x = np.outer(np.outer(h_dash, h), h).reshape(kernel_size, kernel_size, kernel_size)
        kernel_y = np.outer(np.outer(h, h_dash), h).reshape(kernel_size, kernel_size, kernel_size)
        kernel_z = np.outer(np.outer(h, h), h_dash).reshape(kernel_size, kernel_size, kernel_size)

        self.sobel_kernel = torch.from_numpy(
            np.array([kernel_x, kernel_y, kernel_z]).reshape((3, 1, kernel_size, kernel_size, kernel_size))).float().to(
            self.conv_1.weight.device)
        self.sobel_kernel /= self.sobel_kernel.abs().max()


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), inplace=True, negative_slope=1e-2)


class CostRegNet(nn.Module):
    def __init__(self, latent_dim):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(latent_dim, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = ConvBnReLU3D(8, latent_dim)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x
