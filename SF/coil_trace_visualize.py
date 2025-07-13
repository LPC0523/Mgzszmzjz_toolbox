import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import matplotlib as mpl
from einops import rearrange
import nibabel as nib

# Load NIfTI file
img = nib.load('Nifti/SF_map_processing_nii/phj_dic_nii/phj_dic_deltaB0_ppm.nii.gz')
B0_data = img.get_fdata()  # Get the B0 field data
affine = img.affine         # Extract affine matrix for coordinate transformation
dims = B0_data.shape        # Get the dimensions of the data

# Generate voxel indices (i, j, k) for the entire volume
voxel_indices = np.array(np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]), indexing='ij'))
voxel_indices = voxel_indices.reshape(3, -1).T  # Reshape to a list of (i, j, k) coordinates
print(voxel_indices.shape)

# Apply affine transformation to convert voxel indices to world coordinates
voxel_indices_h = np.c_[voxel_indices, np.ones(voxel_indices.shape[0])]  # Add homogeneous coordinate
world_coords = voxel_indices_h.dot(affine.T)                             # Transform to world coordinates
world_coords = world_coords[:, :3]                                       # Keep only X, Y, Z coordinates

# Apply threshold to B0_data
threshold = 1e-4
B0_data[np.abs(B0_data) < threshold] = 0  # Set near-zero values to zero

# Reshape world coordinates to match the B0_data volume dimensions
X = world_coords[:, 0].reshape(dims)
Y = world_coords[:, 1].reshape(dims)
Z = world_coords[:, 2].reshape(dims)

# Save the B0 map and world coordinates
savemat('B0_map_full_phj.mat', {'B0_map': B0_data, 'X': X / 1000, 'Y': Y/ 1000, 'Z': Z / 1000})

# Load saved data for further processing
B0_map = loadmat("B0_map_full_phj.mat")
# data_coil_trace = loadmat('coil_current_optimization/contour_3d_coords.mat')

# # Load coil trace data
# coil_trace = [np.asarray(data_coil_trace['contour_3d_coords'][0][i]) for i in range(43)]

# Reformat B0 data and apply threshold
B0_data = np.flip(rearrange(np.asarray(B0_map['B0_map']), "x y z -> y x z"), axis=0)
B0_data[np.abs(B0_data) < threshold] = 0

# Extract valid positions where B0_data is not zero
vox_pos = np.asarray([B0_map['Y'], B0_map['X'], B0_map['Z']])
B0_data[B0_data == 0] = np.nan  # Set zero values to NaN
valid_idx = np.argwhere(~np.isnan(B0_data))  # Indices of valid data points
B0_data_valid = B0_data[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]]  # Extract valid B0 data

# Get corresponding X, Y, Z coordinates for valid B0 values
valid_pos = vox_pos[:, valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]]

# Save the valid B0 map and coordinates
savemat('B0_map_valid_full_phj.mat', {
    'B0_data': B0_data_valid,
    'X': valid_pos[0],
    'Y': valid_pos[1],
    'Z': valid_pos[2]
})
print(valid_pos.shape, B0_data_valid.shape)

# Append extra points for colorbar display consistency
valid_pos = np.hstack([valid_pos, np.zeros((3, 2))])
B0_data_valid = np.append(B0_data_valid, [-1, 1])

# # Create a 3D plot of coil traces and B0 field data in multiple views
# fig = plt.figure(dpi=250, figsize=(16, 10))

# # Define plot for each view angle
# for idx, (elev, azim) in enumerate([(0, 0), (-90, 0), (0, 90), (45, 45), (-45, 45), (30, -45)]):
#     ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
#     for tr in coil_trace:
#         tr = tr.T
#         ax.plot3D(tr[0], tr[1], tr[2])  # Plot coil traces
#     ax.set_xlim([-0.2, 0.2])
#     ax.set_ylim([-0.2, 0.2])
#     ax.set_zlim([-0.2, 0.2])
#     ax.scatter(valid_pos[0], valid_pos[1], valid_pos[2], alpha=0.02, c=B0_data_valid, cmap='jet')
#     ax.view_init(elev=elev, azim=azim)  # Set viewing angle
#     ax.set_xlabel('X (m)')
#     ax.set_ylabel('Y (m)')
#     ax.set_zlabel('Z (m)')

# # Add colorbar for B0 field data
# fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-2, vmax=2), cmap='jet'),
#              ax=fig.axes, orientation='horizontal', label='Bz off-resonance (Hz)', shrink=0.6)

# # Save plot as image
# plt.savefig('coil_trace.png')
# plt.close()
