import mcubes
import torch
import torch.nn as nn
from lib.embedder import get_embedder
import torch.nn.functional as F
import numpy as np
import cv2
from lib.base import *


class Density(nn.Module):
    def __init__(self, param_init):
        super(Density, self).__init__()
        for p in param_init:
            param = nn.Parameter(torch.tensor(param_init[p]))
            setattr(self, p, param)

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)


class LaplaceDensity(Density):
    def __init__(self, param_init={'beta': 0.1}, beta_min=0.0001):
        super(LaplaceDensity, self).__init__(param_init=param_init)
        self.beta_min = torch.tensor(beta_min).cuda()

    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta


class SingleScaleVolSDF(nn.Module):
    def __init__(self, xyz_min, xyz_max, resolution, latent_dim=12, rgb_latent_dim=64, **kwargs):
        super(SingleScaleVolSDF, self).__init__()
        setattr(self, 'xyz_min', xyz_min)
        setattr(self, 'xyz_max', xyz_max)

        self.latent_dim = latent_dim
        self.rgb_latent_dim = rgb_latent_dim

        self.density = LaplaceDensity()

        self.implicit_network = SDFNetwork(xyz_min=xyz_min, xyz_max=xyz_max, resolution=resolution,
                                           latent_dim=latent_dim, feature_size=rgb_latent_dim)

        self.color_network = ColorNetwork(feature_size=rgb_latent_dim)

    def forward(self, rays_o, rays_d, viewdirs, global_step=None, **render_kwargs):
        ray_pts, mask_outbox, z_vals = self.sample_ray(rays_o, rays_d, is_train=global_step is not None,
                                                       **render_kwargs)

        batch_size, n_samples, _ = ray_pts.shape
        viewdirs = viewdirs.unsqueeze(1).repeat(1, n_samples, 1)[~mask_outbox]
        sdfs = torch.ones((batch_size, n_samples)) * 100
        rgbs = torch.zeros_like(ray_pts)

        sdf_out, feature_out, gradients_out = self.implicit_network(ray_pts[~mask_outbox])

        sdfs[~mask_outbox] = sdf_out.squeeze()

        color_out = self.color_network(ray_pts[~mask_outbox], gradients_out, viewdirs, feature_out)

        rgbs[~mask_outbox] = color_out

        weights, bg_transmittance = self.volume_rendering(sdfs, z_vals)

        rgb = torch.sum(weights[..., None] * rgbs, 1)
        distances = (rays_o[..., None, :] - ray_pts).norm(dim=-1)
        depth = (weights * distances).sum(dim=-1)

        render_result = {
            'rgb': rgb,
            'depth': depth,
        }
        return render_result

    def volume_rendering(self, sdfs, z_vals):
        density_flat = self.density(sdfs)
        density = density_flat.reshape(-1, z_vals.shape[1])

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, dists[..., -2:-1]], dim=-1)
        # torch.tensor([1e10]).unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1), free_energy], dim=-1)
        alpha = 1 - torch.exp(-free_energy)
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))
        fg_transmittance = transmittance[:, :-1]
        weights = alpha * fg_transmittance
        bg_transmittance = transmittance[:, -1]
        return weights, bg_transmittance

    def sample_ray(self, rays_o, rays_d, near, far, is_train=False, **render_kwargs):
        # 1. max number of query points to cover all possible rays
        N_samples = 500
        N_samples_fine = 300

        # 4. sample points on each ray
        z_vals = torch.linspace(0.0, 1.0, N_samples)
        z_vals = near + (far - near) * z_vals[None, :]
        if is_train:
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., 0:1], mids], dim=-1)
            # uniform samples in those intervals
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand  # [n_rays, n_samples]

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]

        # 5. update mask for query points outside bbox
        mask_outbbox = ((self.xyz_min > rays_pts) | (rays_pts > self.xyz_max)).any(dim=-1)
        return rays_pts, mask_outbbox, z_vals

    def extract_geometry(self, resolution, threshold):
        query_func = lambda pts: -self.implicit_network.forward(pts, sdf_only=True)
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
