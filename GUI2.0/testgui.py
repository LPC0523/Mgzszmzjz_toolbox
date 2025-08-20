import sys, os
from datetime import datetime
import numpy as np
import numpy.ma as ma
import nibabel as nib
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.optimize import minimize
from scipy.ndimage import zoom
from style import style
from B0_cal_algorithm import process_dicom_to_deltaB0
from CoilProfile import load_coil_profile, generate_coil_profile
from HardwareDeploy import deploy_to_teensy

# ------------------- Worker Thread -------------------
class WorkerThread(QThread):
    finished = pyqtSignal(object)    # 计算结果
    error = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)   # 只返回数据
        except Exception as e:
            self.error.emit(str(e))

    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

# ------------------- Main Window -------------------
class MainWindow(QMainWindow):
    SLIDER_UPDATE_DELAY = 50  # ms，滑块防抖延迟

    def __init__(self):
        super().__init__()
        self.ui = loadUi('show.ui', self)
        self.setStyleSheet(style)
        # 按钮映射（保持你原来的映射）
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
            "LoadMask": self.load_mask,
        }
        for btn_name, func in button_map.items():
            btn = self.ui.findChild(QPushButton, btn_name)
            if btn: btn.clicked.connect(func)

        # 数据初始化
        self.orig_b0 = None
        self.sim_b0 = None
        self.opt_b0 = None
        self.mag = None
        self.mask = None
        self.B0_array = None
        self.dicom_path = ""
        self.coil_profile = None
        self.selected_loss = "MSE (L2)"

        # slider
        self.slider_axial = self.ui.Axial
        self.slider_coronal = self.ui.Coronal
        self.slider_sagittal = self.ui.Sagittal
        # 防抖 timer
        self._slider_timer = QTimer()
        self._slider_timer.setSingleShot(True)
        self._slider_timer.timeout.connect(self.update_all_views)

        self.slider_axial.valueChanged.connect(self._on_slider_changed)
        self.slider_coronal.valueChanged.connect(self._on_slider_changed)
        self.slider_sagittal.valueChanged.connect(self._on_slider_changed)

        # canvas setup
        self._setup_canvas()

        # colorbar 输入框
        self.MaxNum.editingFinished.connect(self.update_colorbar_range)
        self.MinNumber.editingFinished.connect(self.update_colorbar_range)

    def _on_slider_changed(self, val):
        self._slider_timer.start(self.SLIDER_UPDATE_DELAY)

    def _connect_sliders(self):
        self.slider_axial.valueChanged.connect(lambda val: self.update_all_views())
        self.slider_coronal.valueChanged.connect(lambda val: self.update_all_views())
        self.slider_sagittal.valueChanged.connect(lambda val: self.update_all_views())

    # ------------------- Canvas Setup -------------------
    def _setup_canvas(self):
        def new_canvas(group):
            fig = Figure(figsize=(8,3))
            canvas = FigureCanvas(fig)
            axes = [fig.add_subplot(1,3,i+1) for i in range(3)]
            for ax in axes:
                ax.axis('off')
            group.layout().addWidget(canvas)
            return fig, canvas, axes, [None]*3  # axes, colorbars
        self.ori_fig, self.ori_canvas, self.ori_axes, self.ori_cbars = new_canvas(self.ui.OrignalMapsGroup)
        self.sim_fig, self.sim_canvas, self.sim_axes, self.sim_cbars = new_canvas(self.ui.SimBox)
        self.opt_fig, self.opt_canvas, self.opt_axes, self.opt_cbars = new_canvas(self.ui.OptBox)

    # ------------------- GUI 弹窗 -------------------
    def show_info(self,msg): QMessageBox.information(self,"完成",msg)
    def show_error(self,msg): QMessageBox.critical(self,"错误",msg)

    # ------------------- 线程执行器 -------------------
    def run_in_thread(self, func, args=(), msg_success="", callback=None):
        worker = WorkerThread(func,*args)
        if callback:
            worker.finished.connect(callback)
        else:
            worker.finished.connect(lambda r: self.show_info(msg_success))
        worker.error.connect(self.show_error)
        worker.start()

    # ------------------- B0 Map & Magnitude -------------------
    def load_b0map(self):
        dir_ = QFileDialog.getExistingDirectory(self,"选择 B₀ Map 文件夹")
        if not dir_: return
        ppm_file = None
        for root,_,files in os.walk(dir_):
            for f in files:
                if "deltab0_hertz" in f.lower() and f.lower().endswith(('.nii','.nii.gz')):
                    ppm_file = os.path.join(root,f)
                    break
            if ppm_file: break
        if not ppm_file:
            self.show_error("没有找到 deltaB0.nii")
            return
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
    def load_mask(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择 Mask", filter="*.nii *.nii.gz")
        if not file_path:
            return
        self.mask = nib.load(file_path).get_fdata()

    # ------------------- DICOM & B0 生成 -------------------
    def set_dicom_path(self):
        path = QFileDialog.getExistingDirectory(self,"Select DICOM Directory")
        if path:
            self.dicom_path = path
            self.ui.DicomPath.setText(path)

    def generate_b0map(self):
        if not self.dicom_path:
            self.show_error("请先选择 DICOM 文件夹")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "选择保存 B0 Map 的目录")
        if not output_dir:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nifti_folder = os.path.join(output_dir, f"B0Map_NIfTI_{timestamp}")
        os.makedirs(nifti_folder, exist_ok=True)

        # 主线程直接执行计算
        delta_te = 0.00246
        freq = 42.58 * 5
        process_dicom_to_deltaB0(self.dicom_path, nifti_folder, delta_te, freq)

    # ------------------- Coil Profile -------------------
    def load_coil_profile(self):
        B0_array = load_coil_profile()  # 直接调用已有函数
        if B0_array is not None:
            self.B0_array = B0_array
            self.show_info("Coil Profile 加载完成")
        else:
            self.show_info("未能成功加载 Coil Profile")

    def generate_coil_profile(self):
        """
        调用封装好的 generate_coil_profile 函数，在主线程执行。
        完成后自动更新 self.B0_array 并弹窗提示。
        """
        try:
            # 调用你封装好的函数，它内部会弹窗让用户选择目录、输入通道数等
            result = generate_coil_profile()

            if result is not None:
                self.B0_array = result  # 更新主界面变量
                self.show_info("Coil Profile 生成完成！")
            else:
                self.show_info("操作取消或未生成任何数据。")
        except Exception as e:
            self.show_error(f"生成 Coil Profile 失败：\n{str(e)}")

    # ------------------- Simulate Optimization -------------------
    def configure_obj(self):
        items = ["MSE (L2)", "Max Error", "Peak-to-Peak"]
        item, ok = QInputDialog.getItem(self, "选择优化目标函数", "目标函数:", items, 0, False)
        if ok and item:
            self.selected_loss = item
            self.show_info(f"已选择目标函数：{item}")

    from scipy.ndimage import zoom

    def simulate_opt(self):
        if self.B0_array is None or self.orig_b0 is None or self.mask is None:
            self.show_error("请先加载 Coil Profile、B0 Map 和 Magnitude Mask")
            return

        # ----------------- 数据准备 -----------------
        mask = (self.mask > 0.5).astype(np.float32)
        B0_array = self.B0_array
        B0_target = self.orig_b0

        # ----------------- 如果尺寸不一致，进行重采样 -----------------
        if B0_array.shape[:3] != mask.shape:
            print(f"⚠️ B0_array 与 mask 尺寸不匹配，正在重采样 B0_array...")
            zoom_factors = (
                mask.shape[0] / B0_array.shape[0],
                mask.shape[1] / B0_array.shape[1],
                mask.shape[2] / B0_array.shape[2],
                1  # 通道维不变
            )
            B0_array = zoom(B0_array, zoom_factors, order=1)  # trilinear interpolation
            print("✅ B0_array 重采样完成，shape:", B0_array.shape)

        mask_flat = mask.flatten()
        valid_idx = mask_flat > 0.5
        nonroi_idx = ~valid_idx

        # ----------------- 构建 A 矩阵和 b 向量 -----------------
        A = B0_array.reshape(-1, B0_array.shape[-1])[valid_idx]
        b = B0_target.flatten()[valid_idx]
        A_nonroi = B0_array.reshape(-1, B0_array.shape[-1])[nonroi_idx]

        # ----------------- 优化配置 -----------------
        n_channels = A.shape[1]
        initial_weights = np.ones(n_channels) / n_channels
        current_range = (-3.0, 3.0)
        bounds = [(current_range[0], current_range[1])] * n_channels
        lambda_penalty = 0.05  # 非ROI惩罚权重

        # ----------------- 选择目标函数 -----------------
        loss_dict = {
            "MSE (L2)": lambda w: np.mean((A @ w + b) ** 2) + lambda_penalty * np.mean((A_nonroi @ w) ** 2),
            "Max Error": lambda w: np.max(np.abs(A @ w + b)) + lambda_penalty * np.mean((A_nonroi @ w) ** 2),
            "Peak-to-Peak": lambda w: (np.max(A @ w + b) - np.min(A @ w - b)) + lambda_penalty * np.mean(
                (A_nonroi @ w) ** 2)
        }

        loss_func = loss_dict.get(getattr(self, "selected_loss", "MSE (L2)"))

        # ----------------- 执行优化 -----------------
        try:
            res = minimize(loss_func, initial_weights, bounds=bounds, method='L-BFGS-B')
            shim_currents = res.x
        except Exception as e:
            self.show_error(f"优化失败: {str(e)}")
            return

        # ----------------- 生成优化后的全脑场 -----------------
        B0_compensated = np.tensordot(B0_array, shim_currents, axes=([3], [0]))
        residual_map = B0_target + B0_compensated
        self.sim_b0 = residual_map

        # ----------------- 更新界面 -----------------
        self._refresh_slider_limits()
        self.update_all_views()

        # ----------------- 保存电流结果 -----------------
        save_path = os.path.join(os.getcwd(), "shim_currents.txt")
        with open(save_path, "w") as f:
            for i, val in enumerate(shim_currents):
                f.write(f"ch{i + 1:02d}: {val:.6f} A\n")
        self.show_info(f"优化完成，电流结果已保存到 {save_path}")

    def view_metrix(self):
        print("查看矩阵")

    def deploy_hardware(self):
        if self.sim_b0 is None:
            self.show_error("请先完成模拟优化")
            return
        def task(): return deploy_to_teensy(self.sim_b0)
        self.run_in_thread(task,msg_success="已下发到 Teensy 硬件")

    # ------------------- 更新绘图 -------------------
    def load_opt_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择优化结果文件", "", "NIfTI Files (*.nii *.nii.gz)")
        if not file_path: return
        try:
            self.opt_b0 = nib.load(file_path).get_fdata()
            self._refresh_slider_limits()
            QTimer.singleShot(0, self.update_all_views)
            self.show_info(f"优化结果已加载：\n{file_path}")
        except Exception as e:
            self.show_error(f"加载优化文件失败：\n{e}")

    def update_all_views(self, vmin=None, vmax=None):
        a = self.slider_axial.value() if self.slider_axial else 0
        c = self.slider_coronal.value() if self.slider_coronal else 0
        s = self.slider_sagittal.value() if self.slider_sagittal else 0

        vmin, vmax = self.get_colorbar_limits() if (vmin is None or vmax is None) else (vmin, vmax)

        if self.orig_b0 is not None or self.mag is not None:
            self._draw_panel(self.ori_axes, self.ori_canvas, self.orig_b0, self.mag, a, c, s,
                             "Original", self.ori_cbars, vmin=vmin, vmax=vmax)
        if self.sim_b0 is not None or self.mag is not None:
            self._draw_panel(self.sim_axes, self.sim_canvas, self.sim_b0, self.mag, a, c, s,
                             "Simulated", self.sim_cbars, vmin=vmin, vmax=vmax)
        if self.opt_b0 is not None or self.mag is not None:
            self._draw_panel(self.opt_axes, self.opt_canvas, self.opt_b0, self.mag, a, c, s,
                             "Optimized", self.opt_cbars, vmin=vmin, vmax=vmax)

    # ------------------- 其他绘图与 slider -------------------
    def _draw_panel(self, axes, canvas, vol, mag, a, c, s, title_prefix, cbar_list, alpha=0.7, vmin=-0.01, vmax=0.01):
        def _slice(arr, kind):
            if arr is None: return None
            try:
                if kind == 'sag': return arr[s, :, :].T
                if kind == 'cor': return arr[:, c, :].T
                return arr[:, :, a].T
            except:
                return None

        for idx, ax in enumerate(axes):
            ax.clear()
            name = ["Sagittal", "Coronal", "Axial"][idx]
            kind = name[:3].lower()
            ov = _slice(vol, kind)
            bg = _slice(mag, kind) if mag is not None else None
            ax.axis('off')
            # 仅当 ov 存在时才显示 mag
            if ov is not None:
                if bg is not None:
                    ax.imshow(ma.masked_invalid(bg), cmap='gray', origin='lower')
                overlay = ma.masked_where((ov == 0) | np.isnan(ov), ov)
                im = ax.imshow(overlay, cmap='jet', alpha=alpha, origin='lower', vmin=vmin, vmax=vmax)
            else:
                im = None
                ax.text(0.5, 0.5, f"No {title_prefix} data", ha='center', va='center')

            ax.set_title(f"{title_prefix} - {name} (slice {a if kind == 'axi' else (c if kind == 'cor' else s)})")
            # colorbar
            if im is not None:
                if cbar_list[idx] is None:
                    cbar_list[idx] = canvas.figure.colorbar(im, ax=ax, orientation='vertical', fraction=0.05, pad=0.04)
                else:
                    cbar_list[idx].update_normal(im)

        canvas.draw_idle()

    def _set_sliders_by_shape(self, shape3):
        if not shape3: return
        x,y,z = shape3[:3]
        self.slider_sagittal.setMaximum(max(0,x-1))
        self.slider_coronal.setMaximum(max(0,y-1))
        self.slider_axial.setMaximum(max(0,z-1))

    def _refresh_slider_limits(self):
        for vol in (self.orig_b0,self.sim_b0,self.opt_b0):
            if isinstance(vol,np.ndarray):
                self._set_sliders_by_shape(vol.shape)
                return

    def get_colorbar_limits(self):
        try: vmin = float(self.MinNumber.text())
        except: vmin = -200; self.MinNumber.setText(str(vmin))
        try: vmax = float(self.MaxNum.text())
        except: vmax = 200; self.MaxNum.setText(str(vmax))
        return vmin, vmax

    def update_colorbar_range(self):
        vmin,vmax = self.get_colorbar_limits()
        QTimer.singleShot(0, lambda: self.update_all_views(vmin=vmin,vmax=vmax))

# ------------------- Run App -------------------
if __name__=='__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
