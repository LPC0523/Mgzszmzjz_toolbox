import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

# Set a threshold for small values to be set to zero
threshold = 1e-4

# Load sham and shim NIfTI images
sham_img = nib.load('Nifti/KLE_nii/kle_reference_nii/kle_reference_nii_regis_deltB0_ppm.nii.gz')
shim_img = nib.load('Nifti/KLE_nii/kle_nii/kle_nii_regis_deltB0_ppm.nii.gz')

# Get data arrays from images
sham_data = sham_img.get_fdata()
shim_data = shim_img.get_fdata()

# Set small values to zero for sham data
sham_data[np.abs(sham_data) < threshold] = 0
# Apply Gaussian smoothing to sham data
sham_data = gaussian_filter(sham_data, sigma=0.2)
# Set zero values to NaN for plotting transparency
sham_data[sham_data == 0] = np.nan
# Transpose the sham data to match desired orientation
sham_data = np.transpose(sham_data, (1, 0, 2))

# Set small values to zero for shim data
shim_data[np.abs(shim_data) < threshold] = 0
# Apply Gaussian smoothing to shim data
shim_data = gaussian_filter(shim_data, sigma=0.2)
# Set zero values to NaN for plotting transparency
shim_data[shim_data == 0] = np.nan
# Transpose shim data and limit to 90 slices
shim_data = np.transpose(shim_data, (1, 0, 2))[:, :, :90]

# Calculate difference between shim and sham data, converted to Hz units
diff_data = (shim_data - sham_data) * 42.58 * 5

# Define color normalization for displaying sham and shim data
norm = matplotlib.colors.Normalize(vmin=-2, vmax=2)

# Create figure and subplots for sham and shim maps
fig, axs = plt.subplots(10, 18, figsize=(40, 40), dpi=500, sharex=True, sharey=True)
axs = axs.ravel()  # Flatten subplot array for easy iteration
len_slices = np.size(sham_data, 2)

# Plot each slice of sham and shim data
for i in range(len_slices):
    im = axs[i * 2].imshow(sham_data[:, :, i], cmap='jet', norm=norm)
    axs[i * 2].set_title(f'Sham {i + 1}')
    axs[i * 2].invert_yaxis()
    axs[i * 2].set_axis_off()

    axs[i * 2 + 1].imshow(shim_data[:, :, i], cmap='jet', norm=norm)
    axs[i * 2 + 1].set_title(f'Shim {i + 1}')
    axs[i * 2 + 1].invert_yaxis()
    axs[i * 2 + 1].set_axis_off()

# Add a color bar for the sham and shim images
cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.02]) 
colorbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='ppm')

# Adjust layout and save sham and shim comparison plot
plt.tight_layout(rect=[0, 0.1, 1, 1]) 
plt.savefig('B0_map(ppm).png')
plt.close()

# Define color normalization for the difference data in Hz
fig, axs_diff = plt.subplots(9, 10, figsize=(40, 40), dpi=500, sharex=True, sharey=True)
axs_diff = axs_diff.ravel()
norm = matplotlib.colors.Normalize(vmin=-200, vmax=200)

# Plot each slice of difference data
for i in range(len_slices):
    im = axs_diff[i].imshow(diff_data[:, :, i], cmap='jet', norm=norm)
    axs_diff[i].set_title(f'Diff {i + 1}')
    axs_diff[i].invert_yaxis()
    axs_diff[i].set_axis_off()

# Add a color bar for the difference plot
cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.02]) 
fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='Hz')

# Adjust layout and save difference plot
plt.tight_layout(rect=[0, 0.1, 1, 1]) 
plt.savefig('Measured B0 offset(Hz).png')
plt.close()
