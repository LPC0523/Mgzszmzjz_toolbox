# Mgzszmzjz_toolbox(V1)（没有原理版）
* **匀场常用处理工具及可视化脚本，V1包含B0场图计算、匀场电流优化及可视化、tSNR计算** 
* **大多数程序依赖于调用FSL (https://fsl.fmrib.ox.ac.uk/fsl) 中的工具包，所以在使用本工具包之前请务必下载FSL，Windows下需要通过WSL来运行，具体教程网站中都有详细介绍。即装完WSL把这个文件夹放进去** 
* **匀场电流优化及可视化来源于 (https://github.com/bughht/MRI_System_Design.git) 给HHT递茶orz。**  
* **有一个 Jason 开源的 multi-coil shimming 工具 (https://rflab.martinos.org/index.php?title=Multi-coil_B0_shimming), 文件太大了，有需要多通道匀场的可以了解一下，是可以成功复现的。**

---
## requierments.txt
```
dcm2niix==1.0.20220715
dicom2nifti==2.5.0
einops==0.8.0
fslpy==3.21.1
imageio==2.36.0
importlib-metadata==4.6.4
importlib_resources==6.4.5
matplotlib==3.9.2
nibabel==5.3.2
nipype==1.9.0
numpy==1.26.4
pillow==11.0.0
pydicom==3.0.1
scikit-image==0.24.0
scipy==1.14.1
```
安装依赖环境
```
pip install requirements.txt -r
```
如果有漏的自己pip install一下就行



## 处理文件的逻辑是:创建一个保存Dicom的文件夹和一个保存Nifti的文件夹，在Dicom文件夹下创建一个用作不同实验的子文件夹(比如扫不同序列、不同时间)，如KLE;然后在该子文件夹下创建该批次实验的被试文件夹，如kle,处理B0map的话需要三个文件夹：分别以B0map、Echo1、Echo2为后缀。在Nfti文件夹中，只需创建一个带"_nii"后缀实验名的文件夹即可，后续脚本会自动生成子文件夹。之后类似的操作都是这个处理逻辑
## B0map_cal:包含存放Dicom数据和Nifti数据的文件夹、计算B0场和计算前后匀场差别的算法，**需注意，在运行前需要手动处理Dicom文件夹保留fieldmap序列扫描的图像，联影设备扫出来会得到三个图：相位图B0map、Echo1和Echo2的Magnitude field图，Echo1和Echo2可以当作结构像用于配准**


* **B0_cal_algorithm.py** ：计算B0场的算法,只需修改下面这两个路径，将path指定到该次实验下
```
# Paths for input DICOM and output NIfTI folders
dicom_directory = r'Dicom/KLE'
nifti_folder = r'Nifti/KLE_nii'
```
* **measured_B0_offset**  ：计算前后匀场差别的算法,将以下路径换成后缀为'_regis_deltB0_ppm'的即可，
```
sham_img = nib.load('Nifti/KLE_nii/kle_reference_nii/kle_reference_nii_regis_deltB0_ppm.nii.gz')
shim_img = nib.load('Nifti/KLE_nii/kle_nii/kle_nii_regis_deltB0_ppm.nii.gz')
```

## coil_current_optimization:计算线圈电流匀场的算法，具体可以看jason的MD
## tSNR_cor & tSNR_cal: 计算配准后的EPI图像的平均值除以标准差
**和B0map一样的处理逻辑， 将实验文件夹换成EPI的即可，此外还需要在处理B0map时得到的仿射矩阵(mat结尾的)，最好是哪个做参考就用哪个的仿射矩阵**
* tSNR_cor: 修改路径即可

```
# Define paths for the EPI DICOM directory, output NIfTI directory, and affine matrix file
epi_dicom = r'Dicom/EPI/'
epi_nifti = r'Nifti/EPI_nii/'
affine_matrix = r'Nifti/KLE_nii/kle_nii/kle_nii_regis.mat'
```
* tSNR_cal:将一下路径换成配准后的EPI图像
```
epi_img = nib.load(r'Nifti/EPI_nii/kle_epi_nii/kle_epi_regis_EPI.nii.gz') 
```

### 文件结构
处理前目录结构：
```
├─Dicom
│  └─KLE
│      ├─kle
│      │  ├─b0map_tra_3_B0Map_142317_1303
│      │  ├─b0map_tra_3_Echo1_142316_1301
│      │  └─b0map_tra_3_Echo2_142316_1302
│      └─kle_reference
│          ├─b0map_tra_B0Map_133232_503
│          ├─b0map_tra_Echo1_133230_501
│          └─b0map_tra_Echo2_133230_502
├─Nifti
│  └─KLE_nii
├─B0_cal_algorithm.py
└─measured_B0_offset.py
```

处理后目录结构：
```
├── B0_cal_algorithm.py
├── Measured B0 offset(Hz).png
├── Nifti
│   ├── EPI_nii
│   │   ├── kle_epi_nii
│   │   │   ├── kle_epi_epi_bold_tra_iso1.6mm_3_20241025134352_1401.json
│   │   │   ├── kle_epi_epi_bold_tra_iso1.6mm_3_20241025134352_1401.nii.gz
│   │   │   ├── kle_epi_epi_bold_tra_iso1.6mm_3_20241025134352_1401a.json
│   │   │   ├── kle_epi_epi_bold_tra_iso1.6mm_3_20241025134352_1401a.nii.gz
│   │   │   └── kle_epi_regis_EPI.nii.gz
│   │   └── kle_epi_reference_nii
│   │       ├── kle_epi_reference_epi_bold_tra_iso1.6mm_20241025132124_601.json
│   │       ├── kle_epi_reference_epi_bold_tra_iso1.6mm_20241025132124_601.nii.gz
│   │       ├── kle_epi_reference_epi_bold_tra_iso1.6mm_20241025132124_601a.json
│   │       ├── kle_epi_reference_epi_bold_tra_iso1.6mm_20241025132124_601a.nii.gz
│   │       └── kle_epi_reference_regis_EPI.nii.gz
│   └── KLE_nii
│       ├── kle_nii
│       │   ├── kle_BrainMap.nii.gz
│       │   ├── kle_PhaseMap.nii.gz
│       │   ├── kle_b0map_tra_3_20241025134352_1301.json
│       │   ├── kle_b0map_tra_3_20241025134352_1301.nii.gz
│       │   ├── kle_b0map_tra_3_20241025134352_1302_e2.json
│       │   ├── kle_b0map_tra_3_20241025134352_1302_e2.nii.gz
│       │   ├── kle_b0map_tra_3_20241025134352_1303_ph.json
│       │   ├── kle_b0map_tra_3_20241025134352_1303_ph.nii.gz
│       │   ├── kle_bet.nii.gz
│       │   ├── kle_bet_mask.nii.gz
│       │   ├── kle_deltaB0_hertz.nii.gz
│       │   ├── kle_deltaB0_ppm.nii.gz
│       │   ├── kle_nii_regis.mat
│       │   ├── kle_nii_regis_deltB0_ppm.nii
│       │   ├── kle_nii_regis_deltB0_ppm.nii.gz
│       │   ├── kle_nii_regis_mag.nii.gz
│       │   └── kle_unwrapped_Brain.nii.gz
│       └── kle_reference_nii
│           ├── kle_reference_BrainMap.nii.gz
│           ├── kle_reference_PhaseMap.nii.gz
│           ├── kle_reference_b0map_tra_20241025132124_501.json
│           ├── kle_reference_b0map_tra_20241025132124_501.nii.gz
│           ├── kle_reference_b0map_tra_20241025132124_502_e2.json
│           ├── kle_reference_b0map_tra_20241025132124_502_e2.nii.gz
│           ├── kle_reference_b0map_tra_20241025132124_503_ph.json
│           ├── kle_reference_b0map_tra_20241025132124_503_ph.nii.gz
│           ├── kle_reference_bet.nii.gz
│           ├── kle_reference_bet_mask.nii.gz
│           ├── kle_reference_deltaB0_hertz.nii.gz
│           ├── kle_reference_deltaB0_ppm.nii.gz
│           ├── kle_reference_nii_regis.mat
│           ├── kle_reference_nii_regis_deltB0_ppm.nii
│           ├── kle_reference_nii_regis_deltB0_ppm.nii.gz
│           ├── kle_reference_nii_regis_mag.nii.gz
│           └── kle_reference_unwrapped_Brain.nii.gz
│ 
└─── measured_B0_offset.py
```
