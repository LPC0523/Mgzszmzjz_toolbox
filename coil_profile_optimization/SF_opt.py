# %%
from scipy.linalg import lstsq
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import matplotlib as mpl
# 圆柱尺寸参数(单位：m)
a = 0.14  # 圆柱半径
L = 0.3  # 圆柱高度
h_phi = 0.004  # 方位角步长
h_z = 0.004   # 高度步长
N_phi = int(2 * np.pi * a / h_phi)  # 方位角方向的网格数量
Nz = int(L / h_z)  # 高度方向的网格数量
N = N_phi * Nz
mu_0 = 4 * np.pi * 1e-7

# 生成源点的网格点坐标
grid_points = []
for n in range(1, N_phi * Nz + 1):
    # 计算 alpha 和 beta
    alpha = (n - 1) % N_phi
    beta = (n - alpha - 1) // N_phi + 1
    
    # 计算源点坐标
    x_n = a * np.cos(alpha * h_phi / a)
    y_n = a * np.sin(alpha * h_phi / a)
    z_n = -L / 2 + (2 * beta - 1) / 2 * h_z
    
    # 将坐标添加到网格点列表
    grid_points.append((x_n, y_n, z_n))

# 转换为 numpy 数组
grid_points = np.array(grid_points)
# 获取圆柱表面上网格点的坐标
x_coords = grid_points[:, 0]
y_coords = grid_points[:, 1]
z_coords = grid_points[:, 2]

# 假设目标磁场 b_k (ΔB0)，从你的 fieldmap 中提取目标磁场数据
# 假设 b_target 是一个包含目标点坐标和磁场强度的3D fieldmap
n = 2
B0_map = sio.loadmat("B0_map_valid.mat")
b_k = B0_map['B0_data'].flatten()
x_k = B0_map['X'].flatten()
y_k = B0_map['Y'].flatten()
z_k = B0_map['Z'].flatten()
x_k = x_k[::n]
y_k = y_k[::n]
z_k = z_k[::n]
b_k = b_k[::n] * 5 / 1e6 
fig = plt.figure(dpi=250, figsize=(16, 10))
for idx, (elev, azim) in enumerate([(0, 0), (-90, 0), (0, 90), (45, 45), (-45, 45), (30, -45)]):
    ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
    # 绘制电流源网格点
    ax.scatter(x_coords, y_coords, z_coords, s=1, marker='o', alpha=0.5, label="Current Sources")
    # 绘制目标磁场点并设置颜色
    ax.scatter(x_k, y_k, z_k,s=1, c=b_k, cmap='jet', marker='o', alpha=0.8, label="Target B Field")
    ax.view_init(elev=elev, azim=azim) 
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-2, vmax=2), cmap='jet'),
             ax=fig.axes, orientation='horizontal', label='Bz off-resonance (ppm)', shrink=0.6)

# 计算 r+ 和 r- 的函数
def compute_r_plus_minus(x_k, y_k, z_k, x_n, y_n, z_n, h_z):
    """计算r_+ 和 r_-"""
    r_plus = np.sqrt((x_k - x_n)**2 + (y_k - y_n)**2 + (z_k - (z_n + h_z / 2))**2)
    r_minus = np.sqrt((x_k - x_n)**2 + (y_k - y_n)**2 + (z_k - (z_n - h_z / 2))**2)
    return r_plus, r_minus

# 计算 c_{k,n} 的函数
def compute_c_k_n(x_k, y_k, z_k, x_n, y_n, z_n, h_phi, h_z, a):
    """计算c_{k,n}的值"""
    r_plus, r_minus = compute_r_plus_minus(x_k, y_k, z_k, x_n, y_n, z_n, h_z)
    c_k_n = (mu_0 / (4 * np.pi * a)) * (x_n * x_k + y_n * y_k - a**2) * (1 / r_minus**3 - 1 / r_plus**3) * h_phi
    return c_k_n

# 计算目标磁场的总和
def compute_field_matrix(x_k, y_k, z_k, grid_points, N, h_phi, h_z, a):
    """计算磁场矩阵C"""
    C = np.zeros((x_k.size, len(grid_points)))
    for i in range(x_k.size):
        for n in range(len(grid_points)):
            x_n, y_n, z_n = grid_points[n]
            C[i, n] = compute_c_k_n(x_k[i], y_k[i], z_k[i], x_n, y_n, z_n, h_phi, h_z, a)
    return C
C = compute_field_matrix(x_k.flatten(), y_k.flatten(), z_k.flatten(), grid_points, len(grid_points), h_phi, h_z, a)
# N = len(grid_points)  # 电流源的数量
# psi_n = np.ones(len(grid_points)) 

# def error_function(psi_n, C, b_k):
#     """计算目标场与计算场之间的二次误差"""
#     b_computed = np.dot(C, psi_n)  # 计算磁场
#     error = np.linalg.norm(b_k - b_computed)**2  # 计算二次误差
#     return error
# # 使用最小化函数优化电流强度 psi_n
# result = minimize(error_function, psi_n, args=(C, b_k), 
#                   method='L-BFGS-B', options={'disp': True, 'maxiter': 100})

# # 优化后的电流强度
# optimized_psi_n = result.x
# print(optimized_psi_n)

# psi_n, _, _, _ = lstsq(C, b_k)

# %%
optimized_psi_n = psi_n
# 获取圆柱表面上网格点的坐标
x_coords = grid_points[:, 0]
y_coords = grid_points[:, 1]
z_coords = grid_points[:, 2]

# 可视化：使用 3D 图绘制电流强度在圆柱表面上的分布
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图：每个点的大小或颜色由电流强度决定
sc = ax.scatter(x_coords, y_coords, z_coords, c=optimized_psi_n, s=5, marker='o', cmap='viridis')

# 设置图表标题和标签
ax.set_title("Optimized Current Intensity Distribution on the Cylinder Surface")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# 添加颜色条，表示电流强度的范围
fig.colorbar(sc, label="Current Intensity (ψₙ)")

# # 如果需要显示每个点的电流强度值，可以通过 `ax.text()` 添加
# for i in range(len(grid_points)):
#     ax.text(x_coords[i], y_coords[i], z_coords[i], f'{optimized_psi_n[i]:.2f}', color='black', fontsize=8)

plt.show()

# %%
from scipy.linalg import circulant
t = 1.4e-3 # thickness
kappa = 5.96e7 # electric conductivity

w_col = np.zeros(N_phi)
w_col[0] = 4
w_col[1] = -1
w_col[-1] = -1
W = circulant(w_col)
print(W)

I_phi = np.eye(N_phi)  # N_phi x N_phi 单位矩阵
Zero = np.zeros((N_phi, N_phi))  # N_phi x N_phi 零矩阵

# 构造 block Toeplitz 结构
blocks = []
for i in range(Nz):
    row = []
    for j in range(Nz):
        if i == j:
            row.append(W)
        elif abs(i - j) == 1:
            row.append(-I_phi)
        else:
            row.append(Zero)
    blocks.append(row)

# 组合块矩阵
R = np.block(blocks)
R = (2 / (kappa * t)) * R  # 乘以前面的系数

# %%
def constrcut_gamma(N, N_phi, N_prime):

    Gamma = np.zeros((N, N_prime))

    e1 = np.zeros((N_prime, 1))
    e1[0, 0] = 1
    Gamma[:N_phi, :] += e1.T

    eN_prime = np.zeros((N_prime, 1))
    eN_prime[-1, 0] = 1
    Gamma[-N_phi:, :] += eN_prime.T

    for i in range(N_phi - 1, N - (N_phi - 1)):
        Gamma[i, i - (N_phi - 1)] = 1
    return Gamma

N_prime = N - 2 * (N_phi - 1)
Gamma = constrcut_gamma(N, N_phi, N_prime)
print(Gamma)

# %%
def optimize_phi_lambda(Gamma, R, b, C, lambda_reg):
    """
    计算优化后的 ψ(λ)
    :param Gamma: 约束矩阵 Gamma (N × N')
    :param R: 正则化矩阵 R (N' × N')
    :param C: 线性算子矩阵 C (M × N)
    :param b: 目标场 b (M × 1)
    :param lambda_reg: 正则化参数 λ
    :return: 计算得到的 ψ(λ)
    """

    # 计算 D = [Γ^T (λR + C^T C) Γ]^{-1}
    term = lambda_reg * R + C.T @ C
    D = np.linalg.inv(Gamma.T @ term @ Gamma)
    
    # 计算 ψ(λ) = Γ D Γ^T C^T b
    psi_lambda = Gamma @ D @ Gamma.T @ C.T @ b
    
    return psi_lambda

# %%
print("Gamma shape:", Gamma.shape)
print("R shape:", R.shape) 
print("C.T @ C shape:", (C.T @ C).shape)

# %%
psi_lambda = optimize_phi_lambda(Gamma, R, b_k, C, lambda_reg = 0.1)
print(psi_lambda)

# %%
optimized_psi_n = psi_lambda
# 获取圆柱表面上网格点的坐标
x_coords = grid_points[:, 0]
y_coords = grid_points[:, 1]
z_coords = grid_points[:, 2]

# 可视化：使用 3D 图绘制电流强度在圆柱表面上的分布
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图：每个点的大小或颜色由电流强度决定
sc = ax.scatter(x_coords, y_coords, z_coords, c=optimized_psi_n, s=5, marker='o', cmap='viridis')

# 设置图表标题和标签
ax.set_title("Optimized Current Intensity Distribution on the Cylinder Surface")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# 添加颜色条，表示电流强度的范围
fig.colorbar(sc, label="Current Intensity (ψₙ)")

# # 如果需要显示每个点的电流强度值，可以通过 `ax.text()` 添加
# for i in range(len(grid_points)):
#     ax.text(x_coords[i], y_coords[i], z_coords[i], f'{optimized_psi_n[i]:.2f}', color='black', fontsize=8)

plt.show()


