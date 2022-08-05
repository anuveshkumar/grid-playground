import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.embedder import get_embedder
import numpy as np
from typing import Union


class NeRF(nn.Module):
    def __init__(self, D=5, W=64, d_in=4, d_in_view=3, multires=6, multires_view=4, output_ch=4, skips=[2],
                 use_viewdirs=True):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha, rgb
        else:
            assert False


def get_sphere_intersection(rays_o, rays_d, r=1.0):
    # Input: n_images x 4 x 4 ; n_images x n_rays x 3
    # Output: n_images * n_rays x 2 (close and far) ; n_images * n_rays
    n_pix, _ = rays_d.shape
    ray_cam_dot = torch.matmul(rays_d.unsqueeze(1), rays_o.unsqueeze(-1)).squeeze()
    under_sqrt = ray_cam_dot ** 2 - (rays_o.squeeze().norm(dim=-1) ** 2 - r ** 2)

    under_sqrt = under_sqrt.reshape(-1)
    mask_intersect = under_sqrt > 0

    sphere_intersections = torch.zeros(n_pix, 2).cuda().float()
    sphere_intersections[mask_intersect] = torch.sqrt(under_sqrt[mask_intersect]).unsqueeze(-1) * torch.Tensor(
        [-1, 1]).cuda().float()
    sphere_intersections[mask_intersect] -= ray_cam_dot.reshape(-1)[mask_intersect].unsqueeze(-1)

    sphere_intersections = sphere_intersections.reshape(n_pix, 2)
    # if the ray doesn't intersect with the sphere then the sampling on such rays with start with
    # atleast a distance of ?
    sphere_intersections = sphere_intersections.clamp_min(0.5)
    mask_intersect = mask_intersect.reshape(n_pix)

    return sphere_intersections, mask_intersect


# SH

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh_bases(basis_dim: int, dirs: torch.Tensor):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.
    :param basis_dim: int SH basis dim. Currently, 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions
    :return: torch.Tensor (..., basis_dim)
    """
    result = torch.empty((*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device)
    result[..., 0] = SH_C0
    if basis_dim > 1:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -SH_C1 * y;
        result[..., 2] = SH_C1 * z;
        result[..., 3] = -SH_C1 * x;
        if basis_dim > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = SH_C2[0] * xy;
            result[..., 5] = SH_C2[1] * yz;
            result[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy);
            result[..., 7] = SH_C2[3] * xz;
            result[..., 8] = SH_C2[4] * (xx - yy);

            if basis_dim > 9:
                result[..., 9] = SH_C3[0] * y * (3 * xx - yy);
                result[..., 10] = SH_C3[1] * xy * z;
                result[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy);
                result[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
                result[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy);
                result[..., 14] = SH_C3[5] * z * (xx - yy);
                result[..., 15] = SH_C3[6] * x * (xx - 3 * yy);

                if basis_dim > 16:
                    result[..., 16] = SH_C4[0] * xy * (xx - yy);
                    result[..., 17] = SH_C4[1] * yz * (3 * xx - yy);
                    result[..., 18] = SH_C4[2] * xy * (7 * zz - 1);
                    result[..., 19] = SH_C4[3] * yz * (7 * zz - 3);
                    result[..., 20] = SH_C4[4] * (zz * (35 * zz - 30) + 3);
                    result[..., 21] = SH_C4[5] * xz * (7 * zz - 3);
                    result[..., 22] = SH_C4[6] * (xx - yy) * (7 * zz - 1);
                    result[..., 23] = SH_C4[7] * xz * (xx - 3 * yy);
                    result[..., 24] = SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
    return result


def xyz2equirect(bearings, reso):
    """
    Convert ray direction vectors into equirectangular pixel coordinates.
    Inverse of equirect2xyz.
    Taken from Vickie Ye
    """
    lat = torch.asin(bearings[..., 1])
    lon = torch.atan2(bearings[..., 0], bearings[..., 2])
    x = reso * 2 * (0.5 + lon / 2 / np.pi)
    y = reso * (0.5 - lat / np.pi)
    return torch.stack([x, y], dim=-1)


def world2grid(points, gsz):
    """
    World coordinates to grid coordinates. Grid coordinates are
    normalized to [0, n_voxels] in each side
    :param points: (N, 3)
    :return: (N, 3)
    """
    offset = 0.5 * gsz
    scaling = 0.5 * gsz
    return torch.addcmul(
        offset.to(device=points.device), points, scaling.to(device=points.device)
    )


# Ray-sphere intersector for MSI
class ConcentricSpheresIntersector:

    def __init__(self,
                 size: torch.Tensor,
                 rorigins: torch.Tensor,
                 rdirs: torch.Tensor,
                 rworld_step: torch.Tensor):
        sphere_scaling = 2.0 / size

        origins = (rorigins + 0.5) * sphere_scaling - 1.0
        dirs = rdirs * sphere_scaling
        inorm = 1.0 / dirs.norm(dim=-1)

        self.world_step_scale = rworld_step * inorm
        dirs = dirs * inorm.unsqueeze(-1)

        self.q2a: torch.Tensor = 2 * (dirs * dirs).sum(-1)
        self.qb: torch.Tensor = 2 * (origins * dirs).sum(-1)
        self.f = self.qb.square() - 2 * self.q2a * (origins * origins).sum(-1)
        self.origins = origins
        self.dirs = dirs

    def intersect(self, r: Union[float, torch.Tensor]):
        """
        Find far intersection of all rays with sphere of radius r
        """
        if isinstance(r, torch.Tensor) and (len(r.shape) > 1):
            return self.intersect_broadcasted(r)
        det = self._det(r)
        success_mask = det >= 0
        result = torch.zeros_like(self.q2a)
        result[success_mask] = (-self.qb[success_mask] +
                                torch.sqrt(det[success_mask])) / self.q2a[success_mask]
        return success_mask, result

    def intersect_near(self, r: float):
        """
        Find near intersection of all rays with sphere of radius r
        """
        det = self._det(r)
        success_mask = det >= 0
        result = torch.zeros_like(self.q2a)
        result[success_mask] = (-self.qb[success_mask] -
                                torch.sqrt(det[success_mask])) / self.q2a[success_mask]
        return success_mask, result

    def _det(self, r: float):
        return self.f + 2 * self.q2a * (r * r)

    def intersect_broadcasted(self, r: torch.Tensor):
        det = self._det_broadcasted(r)
        success_mask = det >= 0
        result = torch.zeros_like(det)
        result[success_mask] = (-self.qb.unsqueeze(-1).expand(-1, r.shape[-1])[success_mask] +
                                torch.sqrt(det[success_mask])) / self.q2a.unsqueeze(-1).expand(-1, r.shape[-1])[
                                   success_mask]
        return success_mask, result

    def _det_broadcasted(self, r: float):
        return self.f.unsqueeze(-1) + 2 * self.q2a.unsqueeze(-1) * (r * r)


# ray box intersection
def calc_ray_bbox_intersection(rays_o, rays_d, xyz_min, xyz_max):
    vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
    rate_a = (xyz_max - rays_o) / vec
    rate_b = (xyz_min - rays_o) / vec
    t_min = torch.minimum(rate_a, rate_b).amax(-1)
    t_max = torch.maximum(rate_a, rate_b).amin(-1)

    intersections = torch.stack((t_min, t_max), dim=-1)
    mask_inside = t_max > t_min
    return intersections, mask_inside


# 3d points to cubemap coordinates
def xyz_to_uv(xyz_pts):
    idx = xyz_pts.abs().argmax(dim=-1)
    values = xyz_pts[torch.arange(xyz_pts.shape[0]), idx]
    sign = torch.sign(values)
    values = values.abs()

    mask_x = idx == 0
    mask_y = idx == 1
    mask_z = idx == 2

    holder = torch.zeros_like(xyz_pts)
    holder[..., 0] = 2 * idx + (-sign + 1) / 2

    holder[mask_x, 1] = (-xyz_pts[mask_x, -1] * sign[mask_x]) / values[mask_x]
    holder[mask_x, 2] = xyz_pts[mask_x, -2] / values[mask_x]

    holder[mask_y, 1] = xyz_pts[mask_y, 0] / values[mask_y]
    holder[mask_y, 2] = (-xyz_pts[mask_y, -1] * sign[mask_y]) / values[mask_y]

    holder[mask_z, 1] = (xyz_pts[mask_z, 0] * sign[mask_z]) / values[mask_z]
    holder[mask_z, 2] = xyz_pts[mask_z, -2] / values[mask_z]

    # holder[...,-1] = -1 * holder[...,-1] # opengl
    holder[..., 0] = (holder[..., 0] - 2.5) / 2.5  # normalize index

    return holder
