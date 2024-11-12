import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Load the EPI (echo planar imaging) data from the NIfTI file
epi_img = nib.load(r'Nifti/EPI_nii/kle_epi_nii/kle_epi_regis_EPI.nii.gz') 
epi_data = epi_img.get_fdata()

# Transpose the data array for consistent orientation (adjusting axis order if needed)
epi_data = np.transpose(epi_data, (1, 0, 2, 3))

# Calculate the mean across the last dimension (time) to get the average signal for each voxel
mean_epi = np.mean(epi_data, axis=-1)

# Calculate the standard deviation across the last dimension to measure signal variability for each voxel
std_epi = np.std(epi_data, axis=-1)

# Print the number of time points in the data
print(np.size(epi_data, 3))

# Calculate temporal SNR (tSNR) by dividing mean signal by standard deviation at each voxel, 
# setting 0 where the standard deviation is 0 to avoid division by zero errors
tsnr = np.divide(mean_epi, std_epi, out=np.zeros_like(mean_epi), where=std_epi != 0)

# Normalize color range for display purposes (tSNR values are normalized between 0 and 100 for visualization)
norm = matplotlib.colors.Normalize(vmin=0, vmax=100)

# Create a figure with subplots arranged in a 9x10 grid for displaying each slice's tSNR
fig, axs = plt.subplots(9, 10, figsize=(30, 30), dpi=500, sharex=True, sharey=True)
axs = axs.ravel()  # Flatten the array of subplots to iterate over them

# Get the number of slices along the z-axis
len = np.size(epi_data, 2)

# Plot tSNR for each slice
for i in range(len):
    im = axs[i].imshow(tsnr[:, :, i], cmap='hot', origin='lower', norm=norm)  # Display tSNR with a 'hot' colormap
    axs[i].set_title(f'tSNR {i+1}')  # Set title for each subplot
    axs[i].set_axis_off()  # Hide axis for a cleaner look

# Add a colorbar to the bottom of the figure to indicate tSNR values
cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.02])  # Position colorbar below the subplots
colorbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='tSNR')
colorbar.set_label('tSNR', fontsize=30, fontweight='bold')  # Set colorbar label
colorbar.ax.tick_params(labelsize=20)  # Set colorbar tick size

# Save the figure as an image file
plt.savefig('tSNR.png')
plt.close()
