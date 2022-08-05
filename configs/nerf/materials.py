_base_ = '../default_neus.py'
expname = 'materials'
base_dir = './logs/nerf_synthetic'

data = dict(
    datadir="/home/architect/data/nerf_synthetic/materials",
    dataset_type="blender",
)
