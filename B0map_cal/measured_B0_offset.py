import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

# Set a threshold for small values to be set to zero
threshold = 1e-4

# Load sham and shim NIfTI images
sham_img = nib.load('B0map_cal/czh/czhb0map_default&volume/shim/czh_B0Map_shim_deltaB0_ppm.nii.gz')
shim_img = nib.load('B0map_cal/czh/czhb0map_default&volume/default/czh_B0Map_default_deltaB0_ppm.nii.gz')

sham_data = sham_img.get_fdata()
shim_data = shim_img.get_fdata()

sham_data[np.abs(sham_data) < threshold] = 0
# Apply Gaussian smoothing to sham data
sham_data = gaussian_filter(sham_data, sigma=0.2)
sham_data[sham_data == 0] = np.nan
sham_data = np.transpose(sham_data, (1, 0, 2))

shim_data[np.abs(shim_data) < threshold] = 0
# Apply Gaussian smoothing to shim data
shim_data = gaussian_filter(shim_data, sigma=0.2)
shim_data[shim_data == 0] = np.nan
shim_data = np.transpose(shim_data, (1, 0, 2))[:, :, :90]

# Calculate difference between shim and sham data, converted to Hz units
diff_data = (shim_data - sham_data) * 42.58 * 5

norm = matplotlib.colors.Normalize(vmin=-2, vmax=2)

fig, axs = plt.subplots(10, 18, figsize=(40, 40), dpi=500, sharex=True, sharey=True)
axs = axs.ravel() 
len_slices = np.size(sham_data, 2)

for i in range(len_slices):
    im = axs[i * 2].imshow(sham_data[:, :, i], cmap='jet', norm=norm)
    axs[i * 2].set_title(f'Sham {i + 1}')
    axs[i * 2].invert_yaxis()
    axs[i * 2].set_axis_off()

    axs[i * 2 + 1].imshow(shim_data[:, :, i], cmap='jet', norm=norm)
    axs[i * 2 + 1].set_title(f'Shim {i + 1}')
    axs[i * 2 + 1].invert_yaxis()
    axs[i * 2 + 1].set_axis_off()

cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.02]) 
colorbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='ppm')

plt.tight_layout(rect=[0, 0.1, 1, 1]) 
plt.savefig('B0_map(ppm).png')
plt.close()

fig, axs_diff = plt.subplots(9, 10, figsize=(40, 40), dpi=500, sharex=True, sharey=True)
axs_diff = axs_diff.ravel()
norm = matplotlib.colors.Normalize(vmin=-200, vmax=200)

# Plot each slice of difference data
for i in range(len_slices):
    im = axs_diff[i].imshow(diff_data[:, :, i], cmap='jet', norm=norm)
    axs_diff[i].set_title(f'Diff {i + 1}')
    axs_diff[i].invert_yaxis()
    axs_diff[i].set_axis_off()

cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.02]) 
fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='Hz')

plt.tight_layout(rect=[0, 0.1, 1, 1]) 
plt.savefig('Measured B0 offset(Hz).png')
plt.close()
