import numpy as np
import matplotlib.pyplot as plt

# 加载 npz 文件中的坐标和 u 值
data = np.load("results_poisson_enhanced/results.4900_coords.npz")
coords_values = data["data"]  # shape: (H*W, 3)

x = coords_values[:, 0]
y = coords_values[:, 1]
values = coords_values[:, 2]

# 确定分辨率（例如 64x64 或 128x128）
# 假设 x 和 y 是均匀网格
N = int(np.sqrt(len(x)))  # 假设是 N*N 的网格
assert N * N == len(x), "不是完整方形网格数据"

# reshape 成图像
u_img = values.reshape((N, N))

# 显示图像（注意 y 在 vertical 方向）
plt.imshow(u_img, cmap='jet', origin='lower', extent=[0, 1, 0, 1])
plt.colorbar(label='u(x, y)')
plt.title("Reconstructed u(x,y) from (x, y, u)")
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.show()