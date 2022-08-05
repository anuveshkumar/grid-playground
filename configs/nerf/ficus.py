_base_ = '../default_neus.py'
expname = 'ficus'
base_dir = './logs/nerf_synthetic'

data = dict(
    datadir="/home/architect/data/nerf_synthetic/ficus",
    dataset_type="blender",
)
