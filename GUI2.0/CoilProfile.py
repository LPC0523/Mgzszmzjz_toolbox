from B0_cal_algorithm import process_dicom_to_deltaB0
import re, shutil, nibabel as nib, os
from PyQt5.QtWidgets import QInputDialog, QFileDialog, QMessageBox
import numpy as np
import os
import re
import shutil
import nibabel as nib
import numpy as np
from PyQt5.QtWidgets import QInputDialog, QFileDialog, QMessageBox
def generate_coil_profile():
    """
    生成 Coil Profile 并返回 B0_array (4D: X, Y, Z, n_channels)
    用户可选择源目录和目标目录。
    """
    # ------------------- 用户输入 -------------------
    num_channels, ok = QInputDialog.getInt(
        None, "输入通道数", "请输入线圈通道总数:", value=32, min=1, max=128
    )
    if not ok:
        return None  # 用户取消

    # 生成键名映射
    keys = []
    for band in range((num_channels + 7) // 8):
        for ch in range(min(8, num_channels - band * 8)):
            keys.append(f"b{band}ch{ch}")
    ch_names = [f"ch{i+1}" for i in range(num_channels)]
    rename_dict = dict(zip(keys, ch_names))
    pattern = re.compile(r"(" + "|".join(keys) + r")")

    # 选择源目录
    source_directory = QFileDialog.getExistingDirectory(None, "选择线圈扫描数据源目录")
    if not source_directory:
        return None

    # 选择目标目录
    destination_directory = QFileDialog.getExistingDirectory(None, "选择保存 Coil Profile 的目录")
    if not destination_directory:
        destination_directory = os.path.join(source_directory, "CoilProfile")
        os.makedirs(destination_directory, exist_ok=True)
        print(f"未选择目录，自动生成保存目录: {destination_directory}")

    nifti_folder = destination_directory + "_nii"
    os.makedirs(nifti_folder, exist_ok=True)

    # ------------------- 重命名与移动 -------------------
    def rename_and_move_folders(source_dir, dest_dir):
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        for folder in os.listdir(source_dir):
            folder_path = os.path.join(source_dir, folder)
            if os.path.isdir(folder_path):
                match = pattern.search(folder)
                if match:
                    key = match.group(1)
                    new_name = rename_dict[key]

                    if "0.5n" in folder:
                        current_dir = "negative"
                    elif "0.5p" in folder:
                        current_dir = "positive"
                    else:
                        current_dir = "unknown"

                    new_folder_path = os.path.join(dest_dir, new_name, current_dir, folder)
                    os.makedirs(os.path.dirname(new_folder_path), exist_ok=True)
                    shutil.move(folder_path, new_folder_path)
                    print(f"Moved {folder} -> {new_folder_path}")

    rename_and_move_folders(source_directory, destination_directory)

    # ------------------- DICOM -> NIfTI -------------------
    delta_TE = 0.00246  # Echo time difference in seconds
    Frequency = 42.58 * 5  # 5T

    for dicom_directory in os.listdir(destination_directory):
        ch_dicom_path = os.path.join(destination_directory, dicom_directory)
        ch_nii_path = os.path.join(nifti_folder, f"{dicom_directory}_nii")
        process_dicom_to_deltaB0(ch_dicom_path, ch_nii_path, delta_TE, Frequency)
        print(f"Processed DICOM {ch_dicom_path} -> NIfTI {ch_nii_path}")

    # ------------------- 生成 delta fieldmap -------------------
    B0_list = []
    for ch_nii in os.listdir(nifti_folder):
        ch_nii_path = os.path.join(nifti_folder, ch_nii)
        neg_path = os.path.join(ch_nii_path, "negative_nii")
        pos_path = os.path.join(ch_nii_path, "positive_nii")
        if os.path.exists(neg_path) and os.path.exists(pos_path):
            neg_hertz_fieldmap = sorted([f for f in os.listdir(neg_path) if f.endswith("_hertz.nii.gz")])
            pos_hertz_fieldmap = sorted([f for f in os.listdir(pos_path) if f.endswith("_hertz.nii.gz")])

            for neg_file, pos_file in zip(neg_hertz_fieldmap, pos_hertz_fieldmap):
                neg_img = nib.load(os.path.join(neg_path, neg_file))
                pos_img = nib.load(os.path.join(pos_path, pos_file))

                delta_B0_fieldmap = pos_img.get_fdata() - neg_img.get_fdata()
                delta_img = nib.Nifti2Image(delta_B0_fieldmap, affine=neg_img.affine)
                delta_path = os.path.join(ch_nii_path, f"{ch_nii}_deltaFieldmap.nii.gz")
                nib.save(delta_img, delta_path)
                print(f"Saved delta fieldmap: {delta_path}")
                B0_list.append(delta_B0_fieldmap)

    # ------------------- 堆叠为 4D 数组 -------------------
    B0_array = np.stack(B0_list, axis=-1) if B0_list else None

    QMessageBox.information(None, "完成", "Coil Profile 生成完成！")
    return B0_array



def load_coil_profile():
    """加载已有的 Coil Profile，用于 B0 优化"""
    channel_folder = QFileDialog.getExistingDirectory(None, "选择 Coil Profile 文件夹")
    if not channel_folder:
        return None

    print(f"加载 Coil Profile：{channel_folder}")

    def natural_key(s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

    sorted_channels = sorted(os.listdir(channel_folder), key=natural_key)

    B0_list = []
    missing_channels = []
    for ch_name in sorted_channels:
        ch_dir = os.path.join(channel_folder, ch_name)
        nii_gz_path = os.path.join(ch_dir, f"{ch_name}_deltaFieldmap.nii.gz")
        nii_path = os.path.join(ch_dir, f"{ch_name}_deltaFieldmap.nii")

        if os.path.exists(nii_gz_path):
            ch_path = nii_gz_path
        elif os.path.exists(nii_path):
            ch_path = nii_path
        else:
            missing_channels.append(ch_name)
            print(f"⚠️ 通道 {ch_name} 缺失文件：{nii_gz_path} 或 {nii_path}")
            continue

        B0_ch = nib.load(ch_path).get_fdata()
        B0_list.append(B0_ch)
        print(f"✅ 加载通道 {ch_name}")

    if not B0_list:
        QMessageBox.warning(None, "错误", "没有成功加载任何通道数据！")
        return None

    # 堆叠成 4D 数组 (X,Y,Z,n_channels)
    B0_array = np.stack(B0_list, axis=-1)

    if missing_channels:
        QMessageBox.warning(None, "警告", f"以下通道缺失：{', '.join(missing_channels)}")

    print(f"✅ Coil Profile 加载完成，共 {B0_array.shape[-1]} 个通道")
    return B0_array
