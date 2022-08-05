import argparse
import copy
import os
import random

import imageio
import mmcv
import numpy as np
import torch
import trimesh
from tqdm import tqdm, trange
from lib import utils
from lib.neus import singlescalegrid_volsdf
from lib import ray_processors as rp
from lib.load_data import load_data
import time
import torch.nn.functional as F
import cv2
import shutil


def config_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', default='configs/nerf/drums.py', help='config file path')
    parser.add_argument("--seed", type=int, default=777, help='Random seed')
    parser.add_argument("--resume", action='store_true', default=False)
    parser.add_argument("--test_only", action='store_true', default=False,
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', default=False)
    parser.add_argument("--render_train", action='store_true', default=False)
    parser.add_argument("--render_video", action='store_true', default=False)
    parser.add_argument("--extract_mesh", action='store_true', default=False)

    parser.add_argument("--i_print", type=int, default=100, help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=5000, help='frequency of weight ckpt saving')

    return parser


def render_viewpoints(model, render_poses, HW, Ks, render_kwargs,
                      gt_imgs=None, savedir=None, render_factor=2):
    if render_factor != 0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW //= render_factor
        Ks[:, :2, :3] //= render_factor

    rgbs = []
    rgb_diffs = []
    depths = []
    grads = []
    disps = []
    psnrs = []

    for i, c2w in enumerate(tqdm(render_poses)):
        H, W = HW[i]
        K = Ks[i]
        ray_det = rp.get_rays_of_a_view(H, W, K, c2w, inverse_y=render_kwargs['inverse_y'])
        ray_det = [ray_det[i].cuda().reshape(-1, ray_det[i].shape[-1]) for i in range(len(ray_det))]
        rays_o, rays_d, viewdirs, uv, radii = ray_det
        keys = ['rgb', 'depth', 'disp', 'grads']
        render_results_chunks = [
            {k: v.detach().cpu() for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(1024, 0), rays_d.split(1024, 0), viewdirs.split(1024, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_results_chunks]).reshape(H, W, -1)
            for k in render_results_chunks[0].keys()
        }

        rgb = render_result['rgb'].numpy()
        disp = render_result['disp'].numpy()
        depth = render_result['depth'].numpy()
        grad = render_result['grads'].numpy()
        rgbs.append(rgb)
        disps.append(disp)
        depths.append(depth)
        grads.append(grad)

        rgb_diffs.append(cv2.resize(gt_imgs[i], dsize=(rgb.shape[:2])) - rgb)
        if i == 0:
            print("Testing", rgb.shape, disp.shape, depth.shape)

        if gt_imgs is not None and render_factor == 0:
            p = -10. * np.log10(np.mean(np.square(rgb - gt_imgs[i])))
            psnrs.append(p)

        if savedir is not None:
            rgb8 = utils.to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
            imageio.imwrite(filename.rsplit('.', 1)[0] + '_depth.png', disps[-1])
            imageio.imwrite(filename.rsplit('.', 1)[0] + '_diff.png', rgb_diffs[-1])
            imageio.imwrite(filename.rsplit('.', 1)[0] + '_grad.png', grads[-1])

    rgbs = np.array(rgbs)
    disps = np.array(disps)
    depths = np.array(depths)
    grads = np.array(grads)
    rgb_diffs = np.array(rgb_diffs)

    if len(psnrs):
        print('Testing psnr', np.mean(psnrs), '(avg)')

    return rgbs, rgb_diffs, disps, depths, grads


def seed_everything():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def load_everything(args, cfg):
    data_dict = load_data(cfg.data)
    kept_keys = {'hwf', 'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'irregular_shape', 'poses', 'images',
                 'xyz_min', 'xyz_max', 'render_poses'}

    for k in list(data_dict.keys()):
        if k not in kept_keys:
            data_dict.pop(k)

    if data_dict['irregular_shape']:
        data_dict['images'] = [torch.FloatTensor(im, device='cpu') for im in data_dict['images']]
    else:
        data_dict['images'] = torch.FloatTensor(data_dict['images'], device='cpu')

    data_dict['poses'] = torch.Tensor(data_dict['poses'])
    return data_dict


def train(args, cfg, data_dict):
    print('train: start')
    os.makedirs(os.path.join(cfg.basedir, cfg.expname), exist_ok=True)
    with open(os.path.join(cfg.basedir, cfg.expname, 'args.txt'), 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {} \n'.format(arg, attr))

    cfg.dump(os.path.join(cfg.basedir, cfg.expname, 'config.py'))
    shutil.copy('lib/neus/singlescalegrid_volsdf.py', os.path.join(cfg.basedir, cfg.expname, 'model.py'))
    shutil.copy('train_single_scale_neus.py', os.path.join(cfg.basedir, cfg.expname, 'train.py'))

    xyz_min = data_dict['xyz_min']
    xyz_max = data_dict['xyz_max']

    cfg_model = cfg.model_and_render
    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': 1 if cfg.data.white_bkgd else 0,
        'inverse_y': cfg.data.inverse_y,
        'stepsize': cfg_model.stepsize,
    }

    model_kwargs = copy.deepcopy(cfg_model)
    cfg_train = copy.deepcopy(cfg.train)
    resolution = model_kwargs.pop('resolution')
    if len(cfg_train.pg_scale) and not args.resume:
        resolution = int(resolution / (2 ** len(cfg_train.pg_scale)))

    model = singlescalegrid_volsdf.SingleScaleVolSDF(xyz_min, xyz_max, resolution=resolution, **model_kwargs)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.5e-2)
    model, optimizer, start = load_checkpoint(args, cfg, model, optimizer)

    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, radii_tr, imsz = rp.gather_training_rays(datadict=data_dict, conf=cfg)

    torch.cuda.empty_cache()
    psnr_lst = []
    time0 = time.time()
    global_step = -1

    for global_step in trange(1 + start, 1 + cfg_train.N_iters):
        if global_step % 1000 == 0:
            vertices, triangles = model.extract_geometry(resolution=256, threshold=0.0)
            mesh = trimesh.Trimesh(vertices, triangles)
            mesh.export(os.path.join(cfg.basedir, cfg.expname, '{:0>6d}.ply'.format(global_step)))

        if global_step in cfg_train.pg_scale:
            model.implicit_network.scale_volume_grid(model.implicit_network.resolution * 2)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-2)

        # if global_step % 7000 == 0:
        #     model.implicit_network._finetune()
        #     optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-2)

        if cfg_train.patch_size > 1 and global_step > 7000:
            n_patches = cfg_train.batch_size // (cfg_train.patch_size ** 2)
            patch_size = cfg_train.patch_size
            shape = rgb_tr.shape

            idx_img = np.random.randint(0, shape[0], size=(n_patches, 1))

            x0 = np.random.randint(0, shape[2] - patch_size + 1, size=(n_patches, 1, 1))
            y0 = np.random.randint(0, shape[1] - patch_size + 1, size=(n_patches, 1, 1))
            xy0 = np.concatenate([x0, y0], axis=-1)
            patch_idx = xy0 + np.stack(
                np.meshgrid(np.arange(patch_size), np.arange(patch_size), indexing='xy'),
                axis=-1).reshape(1, -1, 2)
            target = rgb_tr[idx_img, patch_idx[Ellipsis, 1], patch_idx[Ellipsis, 0]].reshape(-1, 3)
            rays_o = rays_o_tr[idx_img, patch_idx[Ellipsis, 1], patch_idx[Ellipsis, 0]].reshape(-1, 3)
            rays_d = rays_d_tr[idx_img, patch_idx[Ellipsis, 1], patch_idx[Ellipsis, 0]].reshape(-1, 3)
            viewdirs = viewdirs_tr[idx_img, patch_idx[Ellipsis, 1], patch_idx[Ellipsis, 0]].reshape(-1, 3)

        else:
            sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.batch_size])
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.batch_size])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.batch_size])
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]

        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)

        render_result = model.forward(rays_o, rays_d, viewdirs, global_step=global_step, **render_kwargs)

        optimizer.zero_grad()
        losses = {}

        losses['rgb'] = cfg_train.weight_main * F.mse_loss(render_result['rgb'], target)
        if cfg_train.patch_size > 1 and global_step > 7000:
            if cfg_train.weight_depth_consistency > 0:
                depth = render_result['depth'].reshape(-1, cfg_train.patch_size, cfg_train.patch_size)
                v00 = depth[:, :-1, :-1]
                v01 = depth[:, :-1, 1:]
                v10 = depth[:, 1:, :-1]
                depth_consistency_loss = (((v00 - v01) ** 2) + ((v00 - v10) ** 2)).mean()
                losses['depth_consistency'] = cfg_train.weight_depth_consistency * depth_consistency_loss
        # losses['eikonal'] = cfg_train.weight_eikonal * (
        #         (torch.linalg.norm(render_result['gradients_deriv'], ord=2, dim=-1) - 1.0) ** 2).mean()
        # losses['normal'] = cfg_train.weight_normal * F.l1_loss(render_result['gradients_deriv'],
        #                                                        F.normalize(render_result['gradients_sobel'], dim=-1))
        psnr = utils.mse2psnr(losses['rgb'].detach()).item()
        # if cfg_train.weight_entropy_last > 0:
        #     pout = render_result['alphainv_cum'][..., -1].clamp(1e-6, 1 - 1e-6)
        #     entropy_last_loss = -(pout * torch.log(pout) + (1 - pout) * torch.log(1 - pout)).mean()
        #     losses['entropy_last'] = cfg_train.weight_entropy_last * entropy_last_loss
        #     calc losses
        # if cfg_train.weight_rgbper > 0:
        #     rgbper = (render_result['raw_rgb'] - target.unsqueeze(-2)).pow(2).sum(-1)
        #     rgbper_Loss = (rgbper * render_result['weights'].detach()).sum(-1).mean()
        #     losses['rgb_per'] = cfg_train.weight_rgbper * rgbper_Loss

        loss = torch.tensor([0.]).to(device)
        for key in losses.keys():
            loss += losses[key]
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e-1)

        optimizer.step()
        psnr_lst.append(psnr)

        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1 / decay_steps)
        optimizer.param_groups[0]['lr'] *= decay_factor
        # check log & save
        if global_step % args.i_print == 0:
            eps_time = time.time() - time0
            eps_time_str = f'{eps_time // 3600:02.0f}:{eps_time // 60 % 60:02.0f}:{eps_time % 60:02.0f}'
            string = f'iter {global_step:6d} / ' \
                     f'Loss: {loss.item():.9f} / PSNR: {np.mean(psnr_lst):5.2f} / ' \
                     f'Eps: {eps_time_str}'

            tqdm.write(string)
            for key in losses.keys():
                losses[key] = losses[key].detach().cpu().item()
            print(losses)
            with open(os.path.join(cfg.basedir, cfg.expname, 'psnrs.txt'), 'a') as f:
                f.write(string + '\n')
            psnr_lst = []

        if global_step % args.i_weights == 0:
            path = os.path.join(cfg.basedir, cfg.expname, f'{global_step:06d}.tar')
            torch.save({
                'global_step': global_step,
                'model_kwargs': model.get_kwargs(),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print(f'scene_rep_reconstruction : saved checkpoints at', path)
        #


def load_checkpoint(args, cfg, model, optimizer):
    # find whether there is existing checkpoint path
    last_ckpt_path = ''
    if args.resume:
        model_list_raw = os.listdir(
            os.path.join(os.path.join(cfg.basedir, cfg.expname)))
        model_list = []
        for model_name in model_list_raw:
            if model_name[-3:] == 'tar':
                model_list.append(model_name)
        model_list.sort()
        latest_model_name = model_list[-1]
        last_ckpt_path = os.path.join(cfg.basedir, cfg.expname, latest_model_name)

    if os.path.isfile(last_ckpt_path):
        reload_ckpt_path = last_ckpt_path
    else:
        reload_ckpt_path = None

    if reload_ckpt_path is None:
        print('training from scratch')
        start = 0
    else:
        print(f'reloading from {reload_ckpt_path}')
        model, optimizer, start = utils.load_checkpoint(model, optimizer, reload_ckpt_path)

    return model, optimizer, start


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    # init environment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    seed_everything()

    data_dict = load_everything(args=args, cfg=cfg)

    if not args.test_only:
        train(args, cfg, data_dict)

    # load model from rendering
    if args.render_test or args.render_train or args.render_video or args.extract_mesh:
        ckpt_path = os.path.join(cfg.basedir, cfg.expname, '010000.tar')

        ckpt_name = ckpt_path.split('/')[-1][:-4]
        model = utils.load_model(singlescalegrid_volsdf.SingleScaleVolSDF, ckpt_path).to(device)

        render_viewpoints_kwargs = {
            'model': model,
            'render_kwargs': {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'inverse_y': cfg.data.inverse_y,
                'stepsize': cfg.model_and_render['stepsize']
            }
        }

    # render trainset and eval
    if args.render_train:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_train_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, rgb_diffs, disps, depths, grads = render_viewpoints(
            render_poses=data_dict['poses'][data_dict['i_train']],
            HW=data_dict['HW'][data_dict['i_train']],
            Ks=data_dict['Ks'][data_dict['i_train']],
            gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_train']],
            savedir=testsavedir,
            **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30,
                         quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.diff.mp4'), utils.to8b(rgb_diffs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.grad.mp4'), utils.to8b(grads), fps=30, quality=8)
    # render testset and eval
    if args.render_test:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_test_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, rgb_diffs, disps, depths, grads = render_viewpoints(
            render_poses=data_dict['poses'][data_dict['i_test']],
            HW=data_dict['HW'][data_dict['i_test']],
            Ks=data_dict['Ks'][data_dict['i_test']],
            gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
            savedir=testsavedir,
            **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30,
                         quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.diff.mp4'), utils.to8b(rgb_diffs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.grad.mp4'), utils.to8b(grads), fps=30, quality=8)

    # render video
    if args.render_video:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, f'render_video_{ckpt_name}')
        os.makedirs(testsavedir, exist_ok=True)
        rgbs, rgb_diffs, disps, depths, grads = render_viewpoints(
            render_poses=data_dict['render_poses'],
            HW=data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
            Ks=data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0),
            savedir=testsavedir,
            **render_viewpoints_kwargs)
        imageio.mimwrite(os.path.join(testsavedir, 'video.rgb.mp4'), utils.to8b(rgbs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.disp.mp4'), utils.to8b(disps / np.max(disps)), fps=30,
                         quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.diff.mp4'), utils.to8b(rgb_diffs), fps=30, quality=8)
        imageio.mimwrite(os.path.join(testsavedir, 'video.grad.mp4'), utils.to8b(grads), fps=30, quality=8)
    print('Done')

    if args.extract_mesh:
        testsavedir = os.path.join(cfg.basedir, cfg.expname, 'mesh_out.ply')
        vertices, triangles = model.extract_geometry(resolution=512, threshold=0.0)

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(cfg.basedir, cfg.expname, 'mesh_out.ply'))
