_base_ = '../default_baseline.py'

expname = 'drums_unet_constantc_biasprelu_rgb_loss_mid_l'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='/home/architect/data/nerf_synthetic/drums',
    dataset_type='blender',
    white_bkgd=False,
)
