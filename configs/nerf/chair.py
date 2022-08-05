_base_ = '../freq_baseline.py'

expname = 'chair'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='/home/neo/Desktop/Anuvesh/data/nerf_synthetic/chair',
    dataset_type='blender',
    white_bkgd=False,
)
