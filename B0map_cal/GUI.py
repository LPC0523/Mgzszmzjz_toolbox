import sys
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog,
    QHBoxLayout, QSlider
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from B0_cal_algorithm import process_dicom_to_deltaB0 

class NiftiViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("一键处理场图 + 三视图可视化")
        self.resize(1200, 800)
        
        self.data = None

        self.button = QPushButton("选择DICOM目录并一键处理")
        self.button.clicked.connect(self.run_full_pipeline)

        self.slider_axial = QSlider(Qt.Horizontal)
        self.slider_coronal = QSlider(Qt.Horizontal)
        self.slider_sagittal = QSlider(Qt.Horizontal)
        for slider in [self.slider_axial, self.slider_coronal, self.slider_sagittal]:
            slider.valueChanged.connect(self.update_plot)

        self.figure, self.axes = plt.subplots(1, 3, figsize=(12, 4))
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.button)
        layout.addWidget(self.canvas)
        layout.addWidget(QLabel("Axial"))
        layout.addWidget(self.slider_axial)
        layout.addWidget(QLabel("Coronal0"))
        layout.addWidget(self.slider_coronal)
        layout.addWidget(QLabel("Sagittal"))
        layout.addWidget(self.slider_sagittal)
        self.setLayout(layout)

    def run_full_pipeline(self):
        dicom_dir = QFileDialog.getExistingDirectory(self, "选择DICOM目录")
        output_dir = QFileDialog.getExistingDirectory(self, "选择Nifti目录")
        if not dicom_dir:
            return
    
        
        delta_te = 0.00246
        freq = 42.58 * 5  # 5T
        process_dicom_to_deltaB0(dicom_dir, output_dir, delta_te, freq)

        ppm_file = None
        for root, _, files in os.walk(output_dir):
            for f in files:
                if "ppm.nii" in f:
                    ppm_file = os.path.join(root, f)
                    break
        if ppm_file is None:
            print("未找到ppm结果")
            return

        self.nii = nib.load(ppm_file)
        self.data = self.nii.get_fdata()
        self.update_sliders()
        self.update_plot()

    def update_sliders(self):
        self.slider_axial.setMaximum(self.data.shape[2] - 1)
        self.slider_coronal.setMaximum(self.data.shape[1] - 1)
        self.slider_sagittal.setMaximum(self.data.shape[0] - 1)

        self.slider_axial.setValue(self.data.shape[2] // 2)
        self.slider_coronal.setValue(self.data.shape[1] // 2)
        self.slider_sagittal.setValue(self.data.shape[0] // 2)

    def update_plot(self):
        if self.data is None:
            return
        a, c, s = self.slider_axial.value(), self.slider_coronal.value(), self.slider_sagittal.value()
        self.axes[0].cla()
        self.axes[0].imshow(self.data[:, :, a].T, cmap='bwr', origin='lower')
        self.axes[0].set_title(f"Axial {a}")

        self.axes[1].cla()
        self.axes[1].imshow(self.data[:, c, :].T, cmap='bwr', origin='lower')
        self.axes[1].set_title(f"Coronal {c}")

        self.axes[2].cla()
        self.axes[2].imshow(self.data[s, :, :].T, cmap='bwr', origin='lower')
        self.axes[2].set_title(f"Sagittal {s}")

        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = NiftiViewer()
    viewer.show()
    sys.exit(app.exec_())
