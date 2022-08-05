_base_ = '../freq_baseline.py'

expname = 'chair'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='/home/architect/data/nerf_synthetic/chair',
    dataset_type='blender',
    white_bkgd=False,
)
