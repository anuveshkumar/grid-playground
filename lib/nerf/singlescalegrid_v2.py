import numpy as np
import torch
import torch.nn as nn

from lib.embedder import get_embedder
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, latent_dim, W=16, rgb_latent_dim=15):
        super(Decoder, self).__init__()
        self.positional_mlp = nn.Sequential(
            nn.Linear(latent_dim, W), nn.ReLU(inplace=True),
            nn.Linear(W, W), nn.ReLU(inplace=True),
            nn.Linear(W, 1 + rgb_latent_dim)
        )

        self.directional_encoder, output_dim = get_embedder(multires=6, input_dims=3, include_inputs=True)
        dim0 = rgb_latent_dim + output_dim
        self.directional_mlp = nn.Sequential(
            nn.Linear(dim0, W), nn.ReLU(inplace=True),
            *[nn.Sequential(nn.Linear(W, W), nn.ReLU(inplace=True))
              for _ in range(2)
              ],
            nn.Linear(W, 3)
        )

    def forward(self, latent, viewdirs):
        positional_out = self.positional_mlp(latent)
        density = positional_out[..., 0:1]
        rgb_latent = positional_out[..., 1:]

        viewdirs_emb = self.directional_encoder(viewdirs)
        rgb_feat = torch.cat([rgb_latent, viewdirs_emb], dim=-1)
        rgb = self.directional_mlp(rgb_feat)

        return torch.cat([density, rgb], dim=-1)


class PriorModules(nn.Module):
    def __init__(self, latent_dim, num_modules):
        super(PriorModules, self).__init__()
        self.attention_network = nn.Sequential(
            nn.Linear(latent_dim + 3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, num_modules),
            nn.Softmax(dim=-1)
        )

        self.module_list = nn.ModuleList([Decoder(latent_dim=latent_dim) for i in range(num_modules)])

    def forward(self, xyz, viewdirs, latents):
        attention = self.attention_network(torch.cat((xyz, latents), dim=-1))
        outs = []
        for module in self.module_list:
            outs.append(module.forward(latents, viewdirs))

        # outs = torch.stack(outs, dim=-2)
        # final_output = (attention[..., None] * outs).sum(dim=-2)
        final_output = attention[..., 1:2] * self.module_list[1].forward(latents, viewdirs)

        return final_output, attention


class SingleScaleGrid(nn.Module):
    def __init__(self,
                 xyz_min=[-1.0] * 3,
                 xyz_max=[1.0] * 3,
                 resolution=256,
                 latent_dim=12,
                 rgb_latent_dim=15,
                 fast_color_thres=0,
                 alpha_init=0,
                 **kwargs):
        super(SingleScaleGrid, self).__init__()

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

        self.grid = torch.nn.Parameter(torch.zeros([1, self.latent_dim, *self.world_size]))
        self.prior_modules = PriorModules(latent_dim=latent_dim, num_modules=4)

    def extract_latent(self, xyz, viewdirs):
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip(-1) * 2 - 1

        latent = F.grid_sample(self.grid, ind_norm, align_corners=True). \
            reshape(self.grid.shape[1], -1).T.reshape(*shape, self.grid.shape[1]).squeeze()
        output, attention = self.prior_modules.forward(ind_norm.squeeze(), viewdirs, latent)

        density, rgb = output[..., 0:1], output[..., 1:]
        return density, rgb, attention

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, **render_kwargs):

        ray_pts, mask_outbox = self.sample_ray(rays_o, rays_d, is_train=global_step is not None, **render_kwargs)
        interval = render_kwargs['stepsize']

        viewdirs = viewdirs.unsqueeze(-2).expand(ray_pts.shape)
        alpha = torch.zeros_like(ray_pts[..., 0])
        rgb = torch.zeros_like(ray_pts)
        grads = torch.zeros_like(ray_pts)

        density, raw_rgb, attention = self.extract_latent(ray_pts[~mask_outbox], viewdirs[~mask_outbox])
        alpha[~mask_outbox] = self.activate_density(density, interval).squeeze()
        rgb[~mask_outbox] = torch.sigmoid(raw_rgb)

        # compute acc transmittance
        weights, alphainv_cum = get_ray_marching_ray(alpha)

        # Ray marching
        rgb_marched = (weights[..., None] * rgb).sum(dim=-2)
        grads_marched = (weights[..., None] * grads).sum(dim=-2)
        depth = (rays_o[..., None, :] - ray_pts).norm(dim=-1)
        depth_marched = (weights * depth).sum(dim=-1)
        acc = weights.sum(dim=-1)

        bg_rgb = render_kwargs['bg']
        bg_depth = render_kwargs['far']

        rgb_marched += alphainv_cum[..., [-1]] * bg_rgb
        depth_marched += alphainv_cum[..., -1] * bg_depth
        disp = 1 / depth_marched
        ret_dict = {
            'alphainv_cum': alphainv_cum,
            'weights': weights,
            'rgb': rgb_marched,
            'raw_rgb': rgb,
            'disp': disp,
            'depth': depth_marched,
            'acc': acc,
            'grads': grads_marched,
            'attention': attention
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
        N_samples = int(np.linalg.norm(np.array(self.world_size.cpu().numpy()) + 1) / stepsize) + 1

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
