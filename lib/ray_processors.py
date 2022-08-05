import time

import numpy as np
import torch


@torch.no_grad()
def gather_training_rays(datadict, conf):
    device = 'cuda:0'
    HW, Ks, near, far, i_train, i_val, i_test, poses, images = [datadict[k] for k in
                                                                ['HW', 'Ks', 'near', 'far', 'i_train', 'i_val',
                                                                 'i_test', 'poses', 'images', ]]
    rgb_tr_ori = images[i_train].to('cpu' if conf.data.load2gpu_on_the_fly else device)

    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, uv_tr, imsz = get_training_rays(
        rgb_tr=rgb_tr_ori, train_poses=poses[i_train], HW=HW[i_train], Ks=Ks[i_train],
        inverse_y=conf.data.inverse_y)
    masks_tr = torch.ones_like(rgb_tr[..., 0]).bool()

    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
@torch.no_grad()
def get_training_rays(rgb_tr, train_poses, HW, Ks, inverse_y):
    print('get_training_rays: start')
    assert len(np.unique(HW, axis=0)) == 1
    assert len(np.unique(Ks.reshape(len(Ks), -1), axis=0)) == 1
    assert len(rgb_tr) == len(train_poses) and len(rgb_tr) == len(Ks) and len(rgb_tr) == len(HW)
    H, W = HW[0]
    K = Ks[0]
    eps_time = time.time()
    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    uv_tr = torch.zeros([len(rgb_tr), H, W, 2], device=rgb_tr.device)
    imsz = [1] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs, uv = get_rays_of_a_view(
            H=H, W=W, K=K, c2w=c2w, inverse_y=inverse_y)
        rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        uv_tr[i].copy_(uv.to(rgb_tr.device))
        del rays_o, rays_d, viewdirs
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, uv_tr, imsz


def get_rays_of_a_view(H, W, K, c2w, inverse_y, mode='center'):
    rays_o, rays_d, uv = get_rays(H, W, K, c2w, inverse_y=inverse_y, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    return rays_o, rays_d, viewdirs, uv


def get_rays(H, W, K, c2w, inverse_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=c2w.device),
        torch.linspace(0, H - 1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    uv = torch.stack((j, i), dim=-1).long()  # height, width
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i + 0.5, j + 0.5
    elif mode == 'random':
        i = i + torch.rand_like(i)
        j = j + torch.rand_like(j)
    else:
        raise NotImplementedError

    if inverse_y:
        dirs = torch.stack([(i - K[0][2]) / K[0][0], (j - K[1][2]) / K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                       -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d, uv
