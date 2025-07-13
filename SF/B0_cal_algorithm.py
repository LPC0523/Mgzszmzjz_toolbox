import os
import numpy as np
import glob
from nipype.interfaces.dcm2nii import Dcm2niix
import nibabel as nib
import gzip
import shutil
from skimage.restoration import unwrap_phase
import subprocess
from fsl.wrappers import bet, flirt, fslmaths


def dcm2nii(dicom_directory, nifti_folder):
    """
    Converts DICOM files in each subdirectory within `dicom_directory` to NIfTI format and saves them in `nifti_folder`.

    Parameters:
        dicom_directory (str): Path to the directory containing DICOM subdirectories.
        nifti_folder (str): Path to the directory where NIfTI files will be saved.
    """
    directories = os.listdir(dicom_directory)
    
    for directory in directories:
        dicom_file_path = os.path.join(dicom_directory, directory)
        nifti_file_path = os.path.join(nifti_folder, f"{directory}_nii")
        
        converter = Dcm2niix()
        converter.inputs.source_dir = dicom_file_path
        converter.inputs.output_dir = nifti_file_path
        
        
        try:
            converter.run()
            print(f"Successfully converted {directory} to NIfTI format.")
        except Exception as e:
            print(f"Error converting {directory} to NIfTI: {e}")


def process_dicom_to_deltaB0(dicom_directory, nifti_folder, delta_TE, Frequency):
    """
    Converts DICOM files to NIfTI format, performs phase unwrapping, applies a brain mask, and calculates delta B0 maps.

    Parameters:
        dicom_directory (str): Path to the directory containing DICOM files.
        nifti_folder (str): Path to the directory where NIfTI files will be saved.
        delta_TE (float): Echo time difference in seconds.
        Frequency (float): Scanner frequency in MHz (e.g., 127.74 MHz for a 3T scanner).
    """
    print("启动！")
    directories = os.listdir(dicom_directory)

    for directory in directories:
        try:
            dicom_file_path = os.path.join(dicom_directory, directory)
            nifti_file_path = os.path.join(nifti_folder, f"{directory}_nii")
            os.makedirs(nifti_file_path, exist_ok=True)

            # Convert DICOM to NIfTI format
            
            converter = Dcm2niix()
            converter.inputs.source_dir = dicom_file_path
            converter.inputs.output_dir = nifti_file_path
            converter.run()
            # Identify phase and magnitude images
            phase_map = glob.glob(os.path.join(nifti_file_path, "*ph.nii.gz*"))
            magnitude_map = glob.glob(os.path.join(nifti_file_path, "*e2.nii.gz*"))

            if phase_map and magnitude_map:
                # Load the phase image and normalize it to range [-π, π]
                phase_map_nifti = nib.load(phase_map[0])
                phase_map_data = phase_map_nifti.get_fdata()
                affine = phase_map_nifti.affine.copy()
                header = phase_map_nifti.header.copy()

                phasemap = phase_map_data / 4096 * np.pi
                phase_map_img_valid = nib.Nifti1Image(phasemap, affine, header)
                phase_map_img_valid_path = os.path.join(nifti_file_path, f"{directory}_PhaseMap.nii.gz")
                nib.save(phase_map_img_valid, phase_map_img_valid_path)

                # Generate brain mask using BET (Brain Extraction Tool)
                bet_path = os.path.join(nifti_file_path, f"{directory}_bet.nii.gz")
                bet(magnitude_map[0], bet_path, f=0.1, m=True)

                # Apply the brain mask to the phase image
                mask_path = os.path.join(nifti_file_path, f"{directory}_bet_mask.nii.gz")
                brain_map_path = os.path.join(nifti_file_path, f"{directory}_BrainMap.nii.gz")
                fslmaths(phase_map_img_valid_path).mul(mask_path).run(brain_map_path)

                # Perform phase unwrapping
                unwrap_path = os.path.join(nifti_file_path, f"{directory}_unwrapped_Brain.nii.gz")
                cmd = [
                    "prelude",
                    "-p", brain_map_path,
                    "-a", magnitude_map[0],
                    "-o", unwrap_path
                ]
                subprocess.run(cmd, check=True)

                # Calculate delta B0 in Hz
                deltB0_hertz_path = os.path.join(nifti_file_path, f"{directory}_deltaB0_hertz.nii.gz")
                fslmaths(unwrap_path).div(2).div(np.pi).div(delta_TE).div(1e6).run(deltB0_hertz_path)

                # Calculate delta B0 in ppm
                deltB0_ppm_path = os.path.join(nifti_file_path, f"{directory}_deltaB0_ppm.nii.gz")
                fslmaths(deltB0_hertz_path).mul(1e6).div(Frequency).run(deltB0_ppm_path)

                print(f"Processing completed for {directory}")
            else:
                print(f"Phase or magnitude images not found for {directory}")
        
        except Exception as e:
            print(f"Error processing {directory}: {e}")



def get_reference_file(nifti_folder):
    """
    Retrieves the reference file (echo2 image) for registration.

    Parameters:
        nifti_folder (str): Path to the directory containing NIfTI files.

    Returns:
        list: Path to the reference echo2 image.
    """
    reference_folder = glob.glob(os.path.join(nifti_folder, "*reference*"))
    if reference_folder:
        reference_img_path = glob.glob(os.path.join(reference_folder[0], "*e2.nii.gz"))
        return reference_img_path if reference_img_path else []
    else:
        print("Reference folder not found.")
        return []


def corigistrator(nifti_folder, reference_file_path):
    """
    Registers each delta B0 map to a reference file for alignment.

    Parameters:
        nifti_folder (str): Path to the directory containing NIfTI files.
        reference_file_path (list): Path to the reference echo2 image.
    """
    if not reference_file_path:
        print("Reference file path not found.")
        return

    directories = os.listdir(nifti_folder)
    
    for directory in directories:
        nii_path = os.path.join(nifti_folder, directory)
        magnitude_file = glob.glob(os.path.join(nii_path, "*e2.nii.gz*"))
        deltB0_map = glob.glob(os.path.join(nii_path, "*ppm*"))

        if magnitude_file and deltB0_map:
            regis_mat = os.path.join(nii_path, f"{directory}_regis.mat")
            regis_mag = os.path.join(nii_path, f"{directory}_regis_mag.nii.gz")
            regis_deltB0_ppm = os.path.join(nii_path, f"{directory}_regis_deltB0_ppm.nii.gz")
            regis_deltB0_ppm_unzipped = regis_deltB0_ppm.replace('.gz', '')

            # Perform rigid registration for the magnitude image
            flirt(src=magnitude_file[0], ref=reference_file_path[0], omat=regis_mat, out=regis_mag)
            # Apply the affine matrix to the delta B0 map
            flirt(src=deltB0_map[0], ref=reference_file_path[0], applyxfm=True, init=regis_mat, out=regis_deltB0_ppm)

            with gzip.open(regis_deltB0_ppm, 'rb') as f_in:
                with open(regis_deltB0_ppm_unzipped, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            print(f"Required images for registration not found for {directory}")
            

# Paths for input DICOM and output NIfTI folders
dicom_directory = r'SF/Dicom/TMS'
nifti_folder = r'SF/Nifti/TMS_nii'

# Parameters for delta B0 calculation
delta_TE = 0.00246  # Echo time difference in seconds
Frequency = 42.58 * 5  # Scanner frequency in MHz for 5T

# Run the main processing and registration functions
process_dicom_to_deltaB0(dicom_directory, nifti_folder, delta_TE, Frequency)
# reference_file_path = get_reference_file(nifti_folder)
# corigistrator(nifti_folder, reference_file_path)

print('Fighting, shut the lab down!')