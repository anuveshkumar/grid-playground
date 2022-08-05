expname = None  # experiment name
basedir = './logs/'  # where to store ckpts and logs

''' Template of data options
'''
data = dict(
    datadir=None,  # path to dataset root folder
    dataset_type=None,  # blender | nsvf | blendedmvs | tankstemple | deepvoxels | co3d
    inverse_y=False,  # intrinsict mode (to support blendedmvs, nsvf, tankstemple)
    load2gpu_on_the_fly=False,  # do not load all images into gpu (to save gpu memory)
    testskip=1,  # subsample testset to preview results
    white_bkgd=False,  # use white background (note that some dataset don't provide alpha and with blended bg color)
    downscale_factor=0,
    factor=4,
)

train = dict(
    N_iters=20000,  # number of optimization steps
    batch_size=2048,  # batch size (number of random rays per optimization step)
    lrate_decay=20,  # lr decay by 0.1 after every lrate_decay*1000 steps
    weight_main=1.0,  # weight of photometric loss
    weight_entropy_last=0.001,  # weight of background entropy loss # 0.01
    weight_sparsity=0.01,
    weight_rgbper=0.01,  # weight of per-point rgb loss # 0.1
    pg_scale=[2000, 3000, 4000, 5500, 7000]  # [1000, 2000, 3000]
)

model_and_render = dict(
    alpha_init=1e-2,  # set the alpha values everywhere at the begin of training
    stepsize=0.5,  # sampling stepsize in volume rendering
)
