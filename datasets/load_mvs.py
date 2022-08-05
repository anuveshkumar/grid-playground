import glob
import os

import numpy as np
from tqdm import tqdm

import re
from PIL import Image
import matplotlib.pyplot as plt
import torch

blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def read_rgb_file(img_path, img_res=None, type='antialias'):
    """
    Read images and resize as per the input image resolution.
    Args:
        img_path: path of img_file
        img_res: tupple of ints for (Width, Height)
        type: 'random', 'patch' or 'importance'- defines the type of ray sampling strategy
        patch_size: (1,1) means stand alone rays and any size greater than this means sampling
            sets of rays in batches of size equal to patch_size.
        N_rays_max: Max no. of rays to be sampled. In case of patch_size (1,1) it is equal
            else it accomodates as many complete patches as possible.
    """
    img = Image.open(img_path)
    # In terms of preserving the quality in PIL:
    # Image.ANTIALIAS > Image.BICUBIC (default) > Image.BILINEAR > Image.NEAREST
    if img_res is not None:
        if type == 'antialias':
            img = img.resize(img_res, Image.ANTIALIAS)
        elif type == 'nearest':
            img = img.resize(img_res, Image.NEAREST)
        elif type == 'bilinear':
            img = img.resize(img_res, Image.BILINEAR)
        else:
            img = img.resize(img_res)  # by default type is Image.BICUBIC
    img = np.array(img) / 255
    return img


def read_mask_file(img_path, img_res=None):
    img = Image.open(img_path)
    img = (np.array(img) / 255).astype(np.long)
    return img


def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size
    # intrinsics[:2, :] /= 4
    return intrinsics, extrinsics


def resize_intrinsics(intrinsics, downsclaed_img_res):
    "img_res in width, height fashion"
    x_scale = (downsclaed_img_res[0] / 2) / intrinsics[0, 2]
    y_scale = (downsclaed_img_res[1] / 2) / intrinsics[1, 2]
    intrinsics[0] = intrinsics[0] * x_scale  # img_res[1] / 1080
    intrinsics[1] = intrinsics[1] * y_scale  # img_res[0] / 1920
    # intrinsics[:2] = intrinsics[:2] / 2 #4, HD - 2
    return intrinsics


def load_mvs_data(basedir):
    basedir = os.path.join(basedir)
    splits = ['train']
    counts = [0]

    rgb_files = sorted(glob.glob(os.path.join(basedir, 'images', '*.jpg')))
    mask_files = sorted(glob.glob(os.path.join(basedir, 'masks', '*.jpg')))
    confidence_mask_files = sorted(glob.glob(os.path.join(basedir, 'mask', '*_final.png')))
    depth_files = sorted([i for i in glob.glob(os.path.join(basedir, "depth_est", "*.pfm")) if 'stage' not in i])
    cam_files = sorted(glob.glob(os.path.join(basedir, "cams", "*.txt")))

    counts.append(len(rgb_files))
    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]
    if len(splits) == 1:
        i_split.append(np.array([]))
        i_split.append(np.array([]))

    img_res = tuple(reversed(read_rgb_file(rgb_files[0]).shape[:2]))
    # img_res = (int(img_res[0] / downscale_factor), int(img_res[1] / downscale_factor))
    print("IMG RES W, H: ", img_res)
    print("num_training_images: ", len(rgb_files))
    poses = []
    K = []
    imgs = []
    for img_file, mask_file, cam_file in tqdm(zip(rgb_files, mask_files, cam_files)):
        img = read_rgb_file(img_file, img_res=img_res)
        mask = read_mask_file(mask_file)
        intrinsic, extrinsic = read_camera_parameters(cam_file)
        extrinsic = np.linalg.inv(extrinsic)
        intrinsic = resize_intrinsics(intrinsic, img_res)

        imgs.append(img * mask[..., None])
        poses.append(extrinsic)
        K.append(intrinsic)

    imgs = np.stack(imgs)
    poses = np.stack(poses)
    K = np.stack(K)

    xyz_min = torch.Tensor([-1.0] * 3)
    xyz_max = - xyz_min
    i_split[0] = np.array([i for i in i_split[0] if i % 2 == 0])
    i_split[1] = i_split[0]
    i_split[2] = i_split[0]
    return imgs, poses, poses, K, i_split, 0.01, 2.0, xyz_min, xyz_max,


if __name__ == "__main__":
    base_dir = "/home/architect/Desktop/HFNeRF/data/chair"
    load_mvs_data(base_dir)
