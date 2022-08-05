_base_ = '../default_neus.py'

expname = 'ship'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='/home/architect/data/nerf_synthetic/ship',
    dataset_type='blender',
)
