import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from lib.embedder import get_embedder
from lib.utils import grid_sample


class SDFNetwork(nn.Module):
    def __init__(self, xyz_min, xyz_max, resolution, latent_dim, feature_size,
                 d_hidden=64, n_layers=2, bias=0.5, scale=1,
                 geometric_init=True, weight_norm=True, inside_outside=False):
        super(SDFNetwork, self).__init__()
        setattr(self, 'xyz_min', xyz_min)
        setattr(self, 'xyz_max', xyz_max)
        self._set_grid_resolution(resolution)

        self.latent_dim = latent_dim
        self.grid = torch.nn.Parameter(torch.zeros([1, self.latent_dim, *self.world_size]).uniform_(-1e-4, 1e-4))
        dims = [latent_dim + 3] + [d_hidden for _ in range(n_layers)] + [1 + feature_size]

        self.num_layers = len(dims)
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            lin = nn.Linear(dims[l], dims[l + 1])

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif latent_dim > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(dims[l + 1]))
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(dims[l + 1]))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs, sdf_only=False):
        latents = self.grid_sampler(inputs, self.grid)
        x = torch.cat([inputs * self.scale, latents], dim=-1)
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        sdf = x[:, :1] / self.scale
        if sdf_only:
            return sdf

        features = x[:, 1:]
        gradients = self.gradient(inputs)
        return sdf, features, gradients

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x, sdf_only=True)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def grid_sampler(self, xyz, *grids, mode=None, align_corners=True):
        '''Wrapper for the interp operation'''
        mode = 'bilinear'
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, 1, -1, 3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        ret_lst = [
            grid_sample(grid, ind_norm)
            .reshape(grid.shape[1], -1).T.reshape(*shape, grid.shape[1]) for grid in grids
        ]
        for i in range(len(grids)):
            if ret_lst[i].shape[-1] == 1:
                ret_lst[i] = ret_lst[i].squeeze(-1)
        if len(ret_lst) == 1:
            return ret_lst[0]
        return ret_lst

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


class ColorNetwork(nn.Module):
    def __init__(self, feature_size, mode='refnerf', d_out=3, d_hidden=64, n_layers=2, weight_norm=True,
                 multires_view=6, squeeze_out=True):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        if mode == 'refnerf':
            d_in = 7
        elif mode == 'idr':
            d_in = 9
        dims = [d_in + feature_size] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view, include_inputs=True, input_dims=3)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

    def forward(self, points, gradients, viewdirs, feature_vectors):
        normals = - (gradients / torch.norm(gradients, dim=-1, keepdim=True))
        refdirs = (2 * (torch.matmul(-viewdirs.unsqueeze(-2), normals.unsqueeze(-1)).squeeze(
            -2)) * normals + viewdirs)
        angle_v_n = torch.matmul(-viewdirs.unsqueeze(-2), normals.unsqueeze(-1)).squeeze(-2)
        if self.embedview_fn is not None and self.mode == 'refnerf':
            refdirs = self.embedview_fn(refdirs)
        elif self.embedview_fn is not None and self.mode == 'idr':
            viewdirs = self.embedview_fn(viewdirs)

        rendering_input = None

        if self.mode == 'idr':
            rendering_input = torch.cat([points, viewdirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'refnerf':
            rendering_input = torch.cat([points, refdirs, angle_v_n, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x
