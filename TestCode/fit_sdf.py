import trimesh
import skimage
import mesh_to_sdf
import mcubes
import numpy as np

mesh = trimesh.load('thai.ply')
voxels = mesh_to_sdf(mesh, 64, pad=True)
np.save('target.npy', voxels)
vertices, triangles = mcubes.marching_cubes(voxels, 0.0)
mesh = trimesh.Trimesh(vertices, triangles)
mesh.export("target.obj")
