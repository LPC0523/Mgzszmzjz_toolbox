import sys
import os
from scipy.optimize import minimize
import numpy as np
from datetime import datetime
from style import gpt_style

import nibabel as nib
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from B0_cal_algorithm import process_dicom_to_deltaB0
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = loadUi('show.ui', self)
        self.setStyleSheet(gpt_style)
        button_map = {
            "LoadB0Map": self.load_b0map,
            "LoadMag": self.load_mag,
            "SetDicomPath": self.set_dicom_path,
            "GenerateB0Map": self.generate_b0map,
            "SimulateOpt": self.simulate_opt,
            "ConfigureObj": self.configure_obj,
            "ViewMetrix": self.view_metrix,
            "DeployHardware": self.deploy_hardware,
            "LoadOptFile": self.load_opt_file,
            "LoadCoilProfile": self.load_coil_profile,
            "GenerateCoilProfile": self.generate_coil_profile,
        }

        self.orig_b0 = None
        self.sim_b0 = None  # 模拟优化后的
        self.opt_b0 = None  # 实测优化后的
        self.mag = None
        self.vmin, self.vmax = -2, 2

        for btn_name, func in button_map.items():
            btn = self.ui.findChild(QPushButton, btn_name)
            if btn:
                btn.clicked.connect(func)
            else:
                print(f"[警告] 找不到按钮 {btn_name}")

        # Original B0
        self.ori_fig = Figure(figsize=(8, 3))
        self.ori_canvas = FigureCanvas(self.ori_fig)
        self.ori_axes = [self.ori_fig.add_subplot(1, 3, i + 1) for i in range(3)]
        self.ui.OrignalMapsGroup.layout().addWidget(self.ori_canvas)

        # Simulated B0
        self.sim_fig = Figure(figsize=(8, 3))
        self.sim_canvas = FigureCanvas(self.sim_fig)
        self.sim_axes = [self.sim_fig.add_subplot(1, 3, i + 1) for i in range(3)]
        self.ui.SimBox.layout().addWidget(self.sim_canvas)

        # Optimized B0
        self.opt_fig = Figure(figsize=(8, 3))
        self.opt_canvas = FigureCanvas(self.opt_fig)
        self.opt_axes = [self.opt_fig.add_subplot(1, 3, i + 1) for i in range(3)]
        self.ui.OptBox.layout().addWidget(self.opt_canvas)

        self.slider_axial = self.ui.Axial
        self.slider_coronal = self.ui.Coronal
        self.slider_sagittal = self.ui.Sagittal
        self.slider_axial.valueChanged.connect(self.update_all_views)
        self.slider_coronal.valueChanged.connect(self.update_all_views)
        self.slider_sagittal.valueChanged.connect(self.update_all_views)

        lineedit_list = ["DicomPath", "MaxNum", "MinNum"]
        self.dicom_path = ""
        for name in lineedit_list:
            le = self.findChild(QLineEdit, name)
            if le:
                func_name = f"on_{name.lower()}_changed"
                if hasattr(self, func_name):
                    le.editingFinished.connect(getattr(self, func_name))

    def load_b0map(self):
        b0_dir = QFileDialog.getExistingDirectory(self, "选择 B₀ Map 文件夹")
        if not b0_dir:
            return

        ppm_file = None
        for root, _, files in os.walk(b0_dir):
            for f in files:
                f_lower = f.lower()
                if "deltb0_ppm" in f_lower and (f_lower.endswith(".nii") or f_lower.endswith(".nii.gz")):
                    ppm_file = os.path.join(root, f)
                    break
            if ppm_file:
                break

        if ppm_file is None:
            print("❌ 没有找到 deltaB0_ppm.nii 或 .nii.gz 文件")
            return

        print("✅ 找到 ppm 文件:", ppm_file)
        self.orig_b0 = nib.load(ppm_file).get_fdata()
        self._refresh_slider_limits()
        self.update_all_views()

    def load_mag(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择 Magnitude 图像", filter="*.nii *.nii.gz")
        if not file_path:
            return
        m = nib.load(file_path).get_fdata()
        mmax = float(np.nanmax(m)) if m is not None else 0.0
        self.mag = (m / mmax) if mmax > 0 else m

        self._refresh_slider_limits()
        self.update_all_views()

    def set_dicom_path(self):
        path = QFileDialog.getExistingDirectory(self, "Select DICOM Directory")
        if path:
            self.dicom_path = path
            self.DicomPath.setText(path)

    def generate_b0map(self):
        """基于已设置的 DICOM 路径生成 B0 Map"""
        if not self.dicom_path:
            QMessageBox.warning(self, "警告", "请先点击 'Set Dicom Path' 选择 DICOM 文件夹！")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "选择保存 B0 Map 的目录")
        if not output_dir:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nifti_folder = os.path.join(output_dir, f"B0Map_NIfTI_{timestamp}")
        os.makedirs(nifti_folder, exist_ok=True)

        try:
            delta_te = 0.00246
            freq = 42.58 * 5  # 5T

            process_dicom_to_deltaB0(self.dicom_path, nifti_folder, delta_te, freq)

            QMessageBox.information(self, "完成", f"B0 Map 已保存到：\n{nifti_folder}")
            print(f"✅ 处理完成：结果保存在 {nifti_folder}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"生成 B0 Map 失败：\n{e}")
            print(f"❌ 错误：{e}")


    def generate_coil_profile(self):
        import re, shutil, nibabel as nib, os
        from PyQt5.QtWidgets import QInputDialog, QFileDialog, QMessageBox

        num_channels, ok = QInputDialog.getInt(
            self, "输入通道数", "请输入线圈通道总数:", value=32, min=1, max=128
        )
        if not ok:
            return  # 用户取消

        keys = []
        for band in range((num_channels + 7) // 8):  # 每8个通道一个 band
            for ch in range(min(8, num_channels - band * 8)):
                keys.append(f"b{band}ch{ch}")
        ch_names = [f"ch{i + 1}" for i in range(num_channels)]
        rename_dict = dict(zip(keys, ch_names))
        pattern = re.compile(r"(" + "|".join(keys) + r")")

        source_directory = QFileDialog.getExistingDirectory(self, "选择线圈扫描数据源目录")
        if not source_directory:
            return

        destination_directory = QFileDialog.getExistingDirectory(self, "选择保存 Coil Profile 的目录")
        if not destination_directory:
            return

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

        nifti_folder = destination_directory + "_nii"
        delta_TE = 0.00246
        Frequency = 42.58 * 5  # 5T

        for dicom_directory in os.listdir(destination_directory):
            ch_dicom_path = os.path.join(destination_directory, dicom_directory)
            ch_nii_path = os.path.join(nifti_folder, f"{dicom_directory}_nii")
            process_dicom_to_deltaB0(ch_dicom_path, ch_nii_path, delta_TE, Frequency)

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

        QMessageBox.information(self, "完成", "Coil Profile 生成完成！")

    def load_coil_profile(self):
        """加载已有的 Coil Profile，用于 B0 优化"""
        channel_folder = QFileDialog.getExistingDirectory(self, "选择 Coil Profile 文件夹")
        if not channel_folder:
            return

        print(f"加载 Coil Profile：{channel_folder}")

        def natural_key(s):
            import re
            return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

        sorted_channels = sorted(os.listdir(channel_folder), key=natural_key)

        B0_list = []
        missing_channels = []

        for ch_name in sorted_channels:
            ch_path = os.path.join(channel_folder, ch_name, f"{ch_name}_deltaFieldmap.nii.gz")
            if os.path.exists(ch_path):
                B0_ch = nib.load(ch_path).get_fdata()
                B0_list.append(B0_ch)
                print(f"✅ 加载通道 {ch_name}")
            else:
                missing_channels.append(ch_name)
                print(f"⚠️ 通道 {ch_name} 缺失文件：{ch_path}")

        if not B0_list:
            QMessageBox.warning(self, "错误", "没有成功加载任何通道数据！")
            return

        # 堆叠成 4D 数组 (X,Y,Z,n_channels)
        B0_array = np.stack(B0_list, axis=-1)
        self.B0_array = B0_array
        self.n_channels = B0_array.shape[-1]

        if missing_channels:
            QMessageBox.warning(self, "警告", f"以下通道缺失：{', '.join(missing_channels)}")

        print(f"✅ Coil Profile 加载完成，共 {self.n_channels} 个通道")

    def configure_obj(self):
        """选择优化目标函数"""
        items = ["MSE (L2)", "Max Error", "Peak-to-Peak"]
        item, ok = QInputDialog.getItem(self, "选择优化目标函数",
                                        "目标函数:", items, 0, False)
        if ok and item:
            self.selected_loss = item
            QMessageBox.information(self, "设置完成", f"已选择目标函数：{item}")
            print(f"✅ 目标函数已设置为: {item}")

    def simulate_opt(self):
        """执行 B0 Shimming 优化模拟并在 GUI 显示结果"""
        if not hasattr(self, 'B0_array') or not hasattr(self, 'B0_target') or not hasattr(self, 'mask'):
            QMessageBox.warning(self, "警告", "请先加载 Coil Profile、目标 B₀ Map 和 mask！")
            return
        if not hasattr(self, 'selected_loss'):
            QMessageBox.warning(self, "警告", "请先点击 '配置 OBJ' 选择目标函数！")
            return

        B0_array = self.B0_array
        B0_target = self.B0_target
        mask = self.mask

        b_flat = B0_target.flatten()
        mask_flat = mask.flatten()
        valid_idx = mask_flat > 0.5
        nonroi_idx = ~valid_idx

        A = B0_array.reshape(-1, B0_array.shape[-1])[valid_idx]
        b = b_flat[valid_idx]
        A_nonroi = B0_array.reshape(-1, B0_array.shape[-1])[nonroi_idx]

        n_channels = A.shape[1]
        initial_weights = np.ones(n_channels) / n_channels
        bounds = [(-3.0, 3.0)] * n_channels
        lambda_penalty = 0.05

        def loss_mse(w):
            return np.mean((A @ w + b) ** 2) + lambda_penalty * np.mean((A_nonroi @ w) ** 2)

        def loss_max(w):
            return np.max(np.abs(A @ w + b)) + lambda_penalty * np.mean((A_nonroi @ w) ** 2)

        def loss_ptp(w):
            r = A @ w - b
            return (r.max() - r.min()) + lambda_penalty * np.mean((A_nonroi @ w) ** 2)

        loss_dict = {
            "MSE (L2)": loss_mse,
            "Max Error": loss_max,
            "Peak-to-Peak": loss_ptp,
        }
        selected_loss_func = loss_dict[self.selected_loss]

        print(f"\n>>> 正在优化目标函数: {self.selected_loss}")
        res = minimize(selected_loss_func, initial_weights, bounds=bounds, method='L-BFGS-B')
        shim_currents = res.x

        # 生成模拟结果体数据（补偿后残差图）
        B0_compensated = np.tensordot(B0_array, shim_currents, axes=([3], [0]))
        residual_map = B0_target + B0_compensated

        # 保存电流
        save_path = os.path.join(os.getcwd(), f"shim_currents_{self.selected_loss.replace(' ', '_')}.txt")
        with open(save_path, "w") as f:
            f.write(f"### {self.selected_loss} ###\nLoss: {selected_loss_func(shim_currents):.6f}\n")
            for i, val in enumerate(shim_currents):
                f.write(f"ch{i + 1:02d}: {val:.6f} A\n")
        print(f"✅ 优化完成，结果已保存到 {save_path}")

        self.sim_b0 = residual_map
        self._refresh_slider_limits()
        self.update_all_views()

        QMessageBox.information(self, "完成", f"优化完成，电流结果已保存：\n{save_path}")

    def view_metrix(self):
        print("查看矩阵")

    def deploy_hardware(self):
        print("部署硬件")

    def load_opt_file(self):
        """加载优化后的真实B0并显示到 Optimized 面板"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择优化结果文件", "", "NIfTI Files (*.nii *.nii.gz)")
        if not file_path:
            return
        try:
            self.opt_b0 = nib.load(file_path).get_fdata()
            self._refresh_slider_limits()
            self.update_all_views()
            QMessageBox.information(self, "完成", f"优化结果已加载：\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载优化文件失败：\n{e}")

    def on_dicompath_changed(self):
        path = self.DicomPath.text()
        print(f"Dicom Path updated: {path}")

    def on_maxnum_changed(self):
        try:
            val = float(self.MaxNum.text())
            print(f"ColorBar Max updated: {val}")
            # 在这里更新 colorbar 最大值
        except ValueError:
            print("无效输入: MaxNum 必须是数字")

    def on_minnumber_changed(self):
        try:
            val = float(self.MinNumber.text())
            print(f"ColorBar Min updated: {val}")
            # 在这里更新 colorbar 最小值
        except ValueError:
            print("无效输入: MinNumber 必须是数字")

    def _set_sliders_by_shape(self, shape3):
        if not shape3:
            return
        x, y, z = shape3[:3]
        self.slider_sagittal.setMaximum(max(0, x - 1))
        self.slider_coronal.setMaximum(max(0, y - 1))
        self.slider_axial.setMaximum(max(0, z - 1))

    def _refresh_slider_limits(self):
        """优先用已加载的任意一块体数据设滑块范围"""
        for vol in (self.orig_b0, self.sim_b0, self.opt_b0):
            if isinstance(vol, np.ndarray):
                self._set_sliders_by_shape(vol.shape)
                return

    def _draw_panel(self, axes, canvas, vol, mag, a, c, s, title_prefix="", alpha=0.7, vmin=-0.01, vmax=0.01):
        """绘制单个 panel（Original / Simulated / Optimized）
        - vol 存在则叠加 jet
        - vol 不存在仅显示 mag 背景
        - mag 不存在或形状不匹配则背景为空
        - 切片标题加索引
        """
        import numpy.ma as ma

        def _slice(arr, kind):
            if arr is None:
                return None
            try:
                if kind == 'sag': return arr[s, :, :].T
                if kind == 'cor': return arr[:, c, :].T
                return arr[:, :, a].T
            except Exception:
                return None

        for ax, name in zip(axes, ["Sagittal", "Coronal", "Axial"]):
            ax.clear()
            idx = {'Sagittal': s, 'Coronal': c, 'Axial': a}[name]
            for ax, name in zip(axes, ["Sagittal", "Coronal", "Axial"]):
                ax.clear()
                idx = {'Sagittal': s, 'Coronal': c, 'Axial': a}[name]

                # 仅当 vol 存在时才显示背景
                ov = _slice(vol, name[:3].lower())
                if ov is not None:
                    # 背景 mag
                    bg = _slice(mag, name[:3].lower()) if mag is not None else None
                    if bg is not None:
                        ax.imshow(ma.masked_invalid(bg), cmap='gray', origin='lower', aspect='auto')

                    # overlay vol
                    overlay = ma.masked_where((ov == 0) | np.isnan(ov), ov)
                    ax.imshow(overlay, cmap='jet', alpha=alpha, origin='lower', vmin=vmin, vmax=vmax, aspect='auto')
                else:
                    ax.text(0.5, 0.5, f"No {title_prefix} data", ha='center', va='center')

                ax.set_title(f"{title_prefix} - {name} (slice {idx})")
                ax.axis('off')

        canvas.draw_idle()

    def update_all_views(self):
        a = self.slider_axial.value()
        c = self.slider_coronal.value()
        s = self.slider_sagittal.value()

        if hasattr(self, "ori_axes"):
            self._draw_panel(self.ori_axes, self.ori_canvas, getattr(self, 'orig_b0', None),
                             getattr(self, 'mag', None), a, c, s, "Original")

        if hasattr(self, "sim_axes"):
            self._draw_panel(self.sim_axes, self.sim_canvas, getattr(self, 'sim_b0', None),
                             getattr(self, 'mag', None), a, c, s, "Simulated")

        if hasattr(self, "opt_axes"):
            self._draw_panel(self.opt_axes, self.opt_canvas, getattr(self, 'opt_b0', None),
                             getattr(self, 'mag', None), a, c, s, "Optimized")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.ui.show()
    sys.exit(app.exec_())
