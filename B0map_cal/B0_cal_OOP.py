import os
import numpy as np
import glob
import gzip
import shutil
import nibabel as nib
from skimage.restoration import unwrap_phase
from nipype.interfaces.dcm2nii import Dcm2niix
from fsl.wrappers import bet, flirt, fslmaths

class DICOMConverter:
    def __init__(self, dicom_dir, nifti_dir):
        self.dicom_dir = dicom_dir
        self.nifti_dir = nifti_dir
        os.makedirs(self.nifti_dir, exist_ok=True)
    
    def convert(self):
        for directory in os.listdir(self.dicom_dir):
            dicom_path = os.path.join(self.dicom_dir, directory)
            nifti_path = os.path.join(self.nifti_dir, f"{directory}_nii")
            os.makedirs(nifti_path, exist_ok=True)
            
            converter = Dcm2niix()
            converter.inputs.source_dir = dicom_path
            converter.inputs.output_dir = nifti_path
            try:
                converter.run()
                print(f"Successfully converted {directory} to NIfTI format.")
            except Exception as e:
                print(f"Error converting {directory} to NIfTI: {e}")


class PhaseProcessor:
    @staticmethod
    def unwrap_and_mask(phase_img_path, mask_img_path, output_path):
        phase_data = nib.load(phase_img_path).get_fdata()
        mask_data = nib.load(mask_img_path).get_fdata()
        unwrapped_phase = unwrap_phase(phase_data) * mask_data
        nib.save(nib.Nifti1Image(unwrapped_phase, nib.load(phase_img_path).affine), output_path)
        return output_path


class B0Calculator:
    def __init__(self, delta_TE, frequency):
        self.delta_TE = delta_TE
        self.frequency = frequency
    
    def calculate_deltaB0(self, unwrapped_path, output_hz_path, output_ppm_path):
        fslmaths(unwrapped_path).div(2 * np.pi * self.delta_TE).run(output_hz_path)
        fslmaths(output_hz_path).mul(1e6).div(self.frequency).run(output_ppm_path)


class MRIRegister:
    @staticmethod
    def register(source, reference, output_mat, output_img):
        flirt(src=source, ref=reference, omat=output_mat, out=output_img)
    
    @staticmethod
    def apply_transform(source, reference, transform_mat, output_img):
        flirt(src=source, ref=reference, applyxfm=True, init=transform_mat, out=output_img)
        
    @staticmethod
    def unzip_gz(file_path):
        unzipped_path = file_path.replace('.gz', '')
        with gzip.open(file_path, 'rb') as f_in:
            with open(unzipped_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return unzipped_path


def main(dicom_dir, nifti_dir, delta_TE, frequency):
    converter = DICOMConverter(dicom_dir, nifti_dir)
    converter.convert()
    
    b0_calc = B0Calculator(delta_TE, frequency)
    
    for directory in os.listdir(nifti_dir):
        nii_path = os.path.join(nifti_dir, directory)
        phase_map = glob.glob(os.path.join(nii_path, "*ph.nii.gz"))
        magnitude_map = glob.glob(os.path.join(nii_path, "*e2.nii.gz"))
        
        if phase_map and magnitude_map:
            bet_path = os.path.join(nii_path, f"{directory}_bet.nii.gz")
            mask_path = os.path.join(nii_path, f"{directory}_bet_mask.nii.gz")
            bet(magnitude_map[0], bet_path, f=0.1, m=True)
            
            brain_map_path = os.path.join(nii_path, f"{directory}_BrainMap.nii.gz")
            fslmaths(phase_map[0]).mul(mask_path).run(brain_map_path)
            
            unwrap_path = os.path.join(nii_path, f"{directory}_unwrapped.nii.gz")
            PhaseProcessor.unwrap_and_mask(brain_map_path, mask_path, unwrap_path)
            
            deltaB0_hz = os.path.join(nii_path, f"{directory}_deltaB0_hz.nii.gz")
            deltaB0_ppm = os.path.join(nii_path, f"{directory}_deltaB0_ppm.nii.gz")
            b0_calc.calculate_deltaB0(unwrap_path, deltaB0_hz, deltaB0_ppm)

    print("Processing complete.")


if __name__ == "__main__":
    dicom_directory = r'./B0map_cal/Dicom/DTI'
    nifti_folder = r'./B0map_cal/Nifti/DTI_nii'
    delta_TE = 0.00246
    frequency = 42.58 * 5  # 5T scanner
    main(dicom_directory, nifti_folder, delta_TE, frequency)
