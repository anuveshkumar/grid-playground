import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.embedder import get_embedder
from lib.pref_embedder_v0 import PREF


class SingleScaleGrid(nn.Module):
    def __init__(self,
                 xyz_min=[-1.0] * 3,
                 xyz_max=[1.0] * 3,
                 alpha_init=0,
                 **kwargs):
        super(SingleScaleGrid, self).__init__()

        self.register_buffer('xyz_min', torch.tensor(xyz_min))
        self.register_buffer('xyz_max', torch.tensor(xyz_max))
        self._set_grid_resolution(resolution=128)
        # determine the density bias shift
        self.alpha_init = alpha_init
        self.act_shift = 0.0
        print('dvgo: set density bias shift to', self.act_shift)
        self.expose = 128

        self.encoder = PREF(res=[160] * 3, d=[8] * 3, ks=32)

        self.density_mlp = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1 + 27)
        )

        # self.feature_encoder = PREF(res=[128] * 3, d=[2] * 3, ks=32)
        # self.feature_mlp = nn.Linear(32, 27, bias=True)

        self.directional_encoder, output_dim = get_embedder(multires=6, input_dims=3, include_inputs=True)
        dim0 = 27 + output_dim
        self.directional_mlp = nn.Sequential(
            nn.Linear(dim0, 128), nn.ReLU(inplace=True),
            *[nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True))
              for _ in range(2)
              ],
            nn.Linear(128, 3)
        )

        self.expose = 128

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, **render_kwargs):

        ray_pts, mask_outbox = self.sample_ray(rays_o, rays_d, is_train=global_step is not None, **render_kwargs)
        interval = render_kwargs['stepsize']

        alpha = torch.zeros_like(ray_pts[..., 0])
        grads = torch.zeros_like(ray_pts)
        encoding = self.encoder.forward(ray_pts[~mask_outbox], bound=self.xyz_max[0].item(),
                                        interp=True)
        # loss = self.density_encoder.parseval_loss()
        density_out = self.density_mlp(encoding)
        density = torch.zeros_like(alpha)
        density[~mask_outbox] = density_out[..., 0].squeeze()
        alpha[~mask_outbox] = self.activate_density(density_out[..., 0], interval).squeeze()

        # features = self.feature_encoder.forward(ray_pts[~mask_outbox], bound=self.xyz_max[0].item(), interp=True)
        # rgb_encoding = self.feature_mlp(features)
        rgb_encoding = density_out[..., 1:]
        # compute acc transmittance
        weights, alphainv_cum = get_ray_marching_ray(alpha)

        viewdirs_emb = self.directional_encoder(viewdirs).unsqueeze(-2).repeat(1, ray_pts.shape[1], 1)[~mask_outbox]
        rgb_feat = torch.cat([rgb_encoding, viewdirs_emb], dim=-1)

        rgb = torch.zeros(*weights.shape, 3).to(weights)

        color_out = self.directional_mlp(rgb_feat)
        rgb[~mask_outbox] = torch.sigmoid(color_out[..., :3])

        # Ray marching
        rgb_marched = (weights[..., None] * rgb).sum(dim=-2)
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
            'grads': grads
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
            "alpha_init": self.alpha_init
        }

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
