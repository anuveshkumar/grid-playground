import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.embedder import get_embedder
from lib.utils import grid_sample, get_sobel_kernel
from lib.nerf.conv_encoder import MultiScaleConvEncoder


class MultiScaleGrid(nn.Module):
    def __init__(self,
                 xyz_min=[-1.0] * 3,
                 xyz_max=[1.0] * 3,
                 resolution=200,
                 latent_dim=16,
                 rgb_latent_dim=15,
                 fast_color_thres=0,
                 alpha_init=0,
                 **kwargs):
        super(MultiScaleGrid, self).__init__()

        self.register_buffer('xyz_min', torch.tensor(xyz_min))
        self.register_buffer('xyz_max', torch.tensor(xyz_max))
        self.latent_dim = latent_dim
        self.rgb_latent_dim = rgb_latent_dim
        self.fast_color_thres = fast_color_thres
        self._set_grid_resolution(resolution)

        # determine the density bias shift
        self.alpha_init = alpha_init
        self.act_shift = np.log(1 / (1 - alpha_init) - 1)
        print('dvgo: set density bias shift to', self.act_shift)

        self.grid_reg = GridReg(rgb_latent_dim=self.rgb_latent_dim)

        self.directional_encoder, output_dim = get_embedder(multires=6, input_dims=3, include_inputs=True)
        dim0 = self.rgb_latent_dim + output_dim
        self.directional_mlp = nn.Sequential(
            nn.Linear(dim0, 64), nn.ReLU(inplace=True),
            *[nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
              for _ in range(2)
              ],
            nn.Linear(64, 3)
        )
        self.saved_grids = None

    def initialize_grids(self):
        self.saved_grids = self.grid_reg.forward(calc_normals=True)

    def extract_latent(self, xyz, calc_normals=False):
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip(-1) * 2 - 1
        if self.saved_grids is None:
            grid_reg, normals = self.grid_reg.forward(calc_normals=calc_normals)
        else:
            grid_reg, normals = self.saved_grids

        latents = grid_sample(grid_reg, ind_norm) \
            .reshape(grid_reg.shape[1], -1).T.reshape(*shape, grid_reg.shape[1]).squeeze()

        if normals is not None:
            normals = F.grid_sample(normals, ind_norm, mode='bilinear', align_corners=True) \
                .reshape(normals.shape[1], -1).T.reshape(*shape, normals.shape[1]).squeeze()
            normals = -F.normalize(normals, dim=-1)
        else:
            normals = torch.zeros_like(xyz)

        return latents, normals

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, **render_kwargs):

        ray_pts, mask_outbox = self.sample_ray(rays_o, rays_d, is_train=global_step is not None, **render_kwargs)
        interval = render_kwargs['stepsize']

        alpha = torch.zeros_like(ray_pts[..., 0])
        normals_sobel = torch.zeros_like(ray_pts)
        normals_deriv = torch.zeros_like(ray_pts)
        positional_out, normal_out = self.extract_latent(ray_pts[~mask_outbox], calc_normals=True)
        alpha[~mask_outbox] = self.activate_density(positional_out[..., 0], interval).squeeze()
        normals_sobel[~mask_outbox] = normal_out

        # Normals
        # rays_pts_temp = ray_pts[~mask_outbox]
        # if len(rays_pts_temp) > 0:
        #     rays_pts_temp.requires_grad_()
        #     pos_out, normal_out = self.extract_latent(rays_pts_temp)
        #     y = self.activate_density(pos_out[..., 0], interval).squeeze()
        #     d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        #     gradients = torch.autograd.grad(
        #         outputs=y,
        #         inputs=rays_pts_temp,
        #         grad_outputs=d_output,
        #         create_graph=True,
        #         retain_graph=True,
        #         only_inputs=True)[0]
        #     normals_deriv[~mask_outbox] = -F.normalize(gradients, dim=-1)

        rgb_latent = torch.zeros(*ray_pts.shape[:2], self.rgb_latent_dim).to(ray_pts.device)
        rgb_latent[~mask_outbox] = positional_out[..., 1:]
        # compute acc transmittance
        weights, alphainv_cum = get_ray_marching_ray(alpha)

        # mask = (weights > self.fast_color_thres)
        # if mask.sum() == 0:
        #     # print('breakpoint')
        #     if global_step is None:
        #         return {
        #             'rgb': torch.zeros((mask.shape[0], 3)).to(rays_o.device),
        #             'disp': torch.zeros((mask.shape[0])).to(rays_o.device),
        #             'depth': torch.zeros((mask.shape[0])).to(rays_o.device)
        #         }

        # viewdirs_emb = self.directional_encoder(viewdirs).unsqueeze(-2).repeat(1, ray_pts.shape[1], 1)[~mask_outbox]

        viewdirs_exp = viewdirs.unsqueeze(-2).expand(normals_sobel.shape)
        refdirs = (2 * (
            torch.matmul(-viewdirs_exp.unsqueeze(-2), normals_sobel.unsqueeze(-1)).squeeze(
                -2)) * normals_sobel + viewdirs_exp)
        angle_v_n = torch.matmul(-viewdirs_exp.unsqueeze(-2), normals_sobel.unsqueeze(-1)).squeeze(-2)[~mask_outbox]
        refdirs_emb = self.directional_encoder(refdirs[~mask_outbox])
        rgb_feat = torch.cat([rgb_latent[~mask_outbox], refdirs_emb], dim=-1)
        rgb = torch.zeros(*weights.shape, 3).to(weights)
        color_out = self.directional_mlp(rgb_feat)
        rgb[~mask_outbox] = torch.sigmoid(color_out[..., :3])

        # Ray marching
        rgb_marched = (weights[..., None] * rgb).sum(dim=-2)
        depth = (rays_o[..., None, :] - ray_pts).norm(dim=-1)
        depth_marched = (weights * depth).sum(dim=-1)
        normals_marched = (weights[..., None] * normals_sobel).sum(dim=-2)
        acc = weights.sum(dim=-1)

        bg_rgb = render_kwargs['bg']
        bg_depth = render_kwargs['far']

        rgb_marched += alphainv_cum[..., [-1]] * bg_rgb
        depth_marched += alphainv_cum[..., -1] * bg_depth
        disp = 1 / depth_marched
        # pts_marched = rays_o + depth_marched[:, None] * viewdirs
        ret_dict = {
            'alphainv_cum': alphainv_cum,
            'weights': weights,
            'rgb': rgb_marched,
            'raw_rgb': rgb,
            'disp': disp,
            'depth': depth_marched,
            'acc': acc,
            'grads': normals_marched,
            'normals_deriv': normals_deriv,
            'normals_sobel': normals_sobel,
            # 'pts': pts_marched
        }
        for key in ret_dict.keys():
            if torch.isnan(ret_dict[key]).sum() > 0:
                print("checkpoint")

        return ret_dict

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        return 1 - torch.exp(-F.softplus(density + self.act_shift) * interval)

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, is_train=False, **render_kwargs):
        # 1. max number of query points to cover all possible rays
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
        return rays_pts, mask_outbbox

    def get_kwargs(self):
        return {
            "xyz_min": self.xyz_min,
            "xyz_max": self.xyz_max,
            "resolution": self.resolution,
            "fast_color_thres": self.fast_color_thres,
            "alpha_init": self.alpha_init
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

    def _set_grid_resolution(self, resolution):
        self.resolution = resolution
        self.world_size = torch.tensor([resolution] * 3).long()
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / (resolution ** 3)).pow(1 / 3)


def cumprod_exclusive(p):
    # Not sure why: it will be slow at the end of training if clamping at 1e-10 is not applied
    return torch.cat([torch.ones_like(p[..., [0]]), p.clamp_min(1e-10).cumprod(-1)], -1)


def get_ray_marching_ray(alpha):
    alphainv_cum = cumprod_exclusive(1 - alpha)
    weights = alpha * alphainv_cum[..., :-1]
    return weights, alphainv_cum


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
