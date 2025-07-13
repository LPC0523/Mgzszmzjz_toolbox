import os 
import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nipype.interfaces.dcm2nii import Dcm2niix
from fsl.wrappers import flirt

def Flirt_cor(dicom_directory, nifti_folder, affinemartix):
    """
    Converts DICOM files in each subdirectory within `dicom_directory` to NIfTI format, saves them in `nifti_folder`, 
    and registers each converted image to a reference image using the affine matrix provided.

    Parameters:
        dicom_directory (str): Path to the directory containing DICOM subdirectories.
        nifti_folder (str): Path to the directory where NIfTI files will be saved.
        affinemartix (str): Path to the affine matrix file used for registration.
    """
    # List all subdirectories in the DICOM directory
    directories = os.listdir(dicom_directory)
    
    for directory in directories:
        # Set paths for each subdirectory
        dicom_file_path = os.path.join(dicom_directory, directory)
        nifti_file_path = os.path.join(nifti_folder, f"{directory}_nii")
        
        # Create output directory if it does not exist
        os.makedirs(nifti_file_path, exist_ok=True)
        
        # Initialize DICOM to NIfTI converter
        converter = Dcm2niix()
        converter.inputs.source_dir = dicom_file_path  # Set source directory for DICOM files
        converter.inputs.output_dir = nifti_file_path  # Set output directory for NIfTI files
        converter.run()
        print(f"Successfully converted {directory} to NIfTI format.")

        # Define paths for further processing
        EPI_fold_path = os.path.join(nifti_folder, f"{directory}_nii")
        EPI_file = glob.glob(os.path.join(EPI_fold_path, "*nii.gz*"))
        regis_EPI = os.path.join(EPI_fold_path, f"{directory}_regis_EPI.nii.gz")
        
        # Get reference file for registration
        reference = get_reference_file(nifti_folder)
        
        # Apply the affine matrix transformation to the EPI file if a reference file is available
        if reference:
            flirt(src=EPI_file[0], ref=reference[0], applyxfm=True, init=affinemartix, out=regis_EPI)

def get_reference_file(nifti_folder):
    """
    Retrieves the reference file (EPI image) for registration. Assumes reference files are stored in subdirectories 
    with names containing "reference".

    Parameters:
        nifti_folder (str): Path to the directory containing NIfTI files.

    Returns:
        list: Path to the reference EPI image, or an empty list if no reference image is found.
    """
    # Look for subdirectory containing "reference" in its name
    reference_folder = glob.glob(os.path.join(nifti_folder, "*reference*"))
    
    # If a reference folder is found, look for the reference NIfTI image inside it
    if reference_folder:
        reference_img_path = glob.glob(os.path.join(reference_folder[0], "*nii.gz"))
        return reference_img_path if reference_img_path else []
    else:
        print("Reference folder not found.")
        return []        

# Define paths for the EPI DICOM directory, output NIfTI directory, and affine matrix file
epi_dicom = r'Dicom/EPI/'
epi_nifti = r'Nifti/EPI_nii/'
affine_matrix = r'Nifti/KLE_nii/kle_nii/kle_nii_regis.mat'

Flirt_cor(epi_dicom, epi_nifti, affine_matrix)
