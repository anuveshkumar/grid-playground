_base_ = '../freq_baseline.py'

expname = 'lego'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='/home/neo/Desktop/Anuvesh/data/nerf_synthetic/lego',
    dataset_type='blender',
    white_bkgd=False,
)
