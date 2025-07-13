import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
import trimesh
import scipy.io as sio
from bfieldtools.mesh_conductor import MeshConductor
from bfieldtools.coil_optimize import optimize_streamfunctions


# Load simple plane mesh that is centered on the origin
faceloop = trimesh.load_mesh(r"inner.STL")

rotation_matrix = np.array(
    [
    [ 1,  0,  0, 0],
    [ 0,  -1,  0, 0],
    [ 0,  0,  -1, 0],
    [ 0,  0,  0, 1]
    ]
)
faceloop.apply_transform(rotation_matrix)
coil = MeshConductor(mesh_obj=faceloop, fix_normals=True)
MeshConductor.plot_mesh(coil)

B0_map = sio.loadmat("B0_map_valid.mat")

b_k = B0_map['B0_data'].flatten() * 5 / 1e6
x_k = B0_map['X'].flatten() * 1000
y_k = B0_map['Y'].flatten() * 1000
z_k = B0_map['Z'].flatten() * 1000
target_points = np.stack([x_k, y_k, z_k], axis=1)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

mesh_verts = faceloop.vertices
ax.scatter(mesh_verts[:, 0], mesh_verts[:, 1], mesh_verts[:, 2], s=1, c='gray', label='Mesh Vertices')

ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2], s=10, c=b_k, cmap = 'jet', label='Target Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.tight_layout()
plt.show()


target_field = b_k.reshape(-1, 1) 
target_abs_error = np.ones_like(target_field) * 1e-6

sample_rate = 500
target_points = target_points[::sample_rate]
target_field = target_field[::sample_rate]
target_abs_error = target_abs_error[::sample_rate]

target_spec = {
    "coupling": coil.B_coupling(target_points)[:, 2, :], 
    "abs_error": target_abs_error,
    "target": target_field,
}

coil.s, prob = optimize_streamfunctions(
    coil,
    [target_spec],
    objective="minimum_inductive_energy",
    solver="SCS"
)


import pickle

# 保存
with open("coil_s_result.pkl", "wb") as f:
    coil = pickle.load(coil.s, f)

coil.s.plot(ncolors=256)