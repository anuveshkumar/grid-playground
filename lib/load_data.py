import numpy as np

from datasets.load_blender import load_blender_data
from datasets.load_mvs import load_mvs_data


def load_data(args):
    K, depths, confidence_masks = None, None, None

    if args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split, xyz_min, xyz_max = load_blender_data(
            args.datadir, args.downscale_factor)

        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_test, i_val = i_split

        near, far = 2.0, 6.0

        if images.shape[-1] == 4:
            if args.white_bkgd:
                images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
            else:
                images = images[..., :3] * images[..., -1:]

    if args.dataset_type == 'mvs':
        images, poses, render_poses, K, i_split, near, far, xyz_min, xyz_max = load_mvs_data(args.datadir)
        i_train, i_test, i_val = i_split

    HW = np.array([im.shape[:2] for im in images])
    irregular_shape = (images.dtype is np.dtype('object'))

    if K is None:
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    # render_poses = render_poses[..., :4]

    data_dict = dict(
        HW=HW, Ks=Ks, near=near, far=far, i_train=i_train, i_val=i_val, i_test=i_test, poses=poses,
        images=images, irregular_shape=irregular_shape, xyz_min=xyz_min, xyz_max=xyz_max, render_poses=render_poses
    )

    return data_dict
