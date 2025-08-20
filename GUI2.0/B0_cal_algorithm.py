# ====== 基础库 ======
import os
import re
import glob
import numpy as np
from scipy.io import loadmat, savemat
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize
import subprocess
import nibabel as nib                          # 处理 NIfTI 文件
from fsl.wrappers import flirt, fslmaths, bet      # FSL 工具包，配准与图像运算
from nipype.interfaces.dcm2nii import Dcm2niix
# ====== 可视化 ======
import matplotlib.pyplot as plt
import matplotlib as mpl
import ipywidgets as widgets
from IPython.display import display

# ====== 高维数据操作（如 patch 操作） ======
from einops import rearrange

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
                fslmaths(unwrap_path).div(2).div(np.pi).div(delta_TE).run(deltB0_hertz_path)

                # Calculate delta B0 in ppm
                deltB0_ppm_path = os.path.join(nifti_file_path, f"{directory}_deltaB0_ppm.nii.gz")
                fslmaths(deltB0_hertz_path).div(Frequency).run(deltB0_ppm_path)

                print(f"Processing completed for {directory}")
            else:
                print(f"Phase or magnitude images not found for {directory}")
        
        except Exception as e:
            print(f"Error processing {directory}: {e}")