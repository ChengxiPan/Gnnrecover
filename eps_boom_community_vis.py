from matplotlib import pyplot as plt
import json
from tqdm import tqdm
import numpy as np
from scipy.interpolate import make_interp_spline

file_path = './analysis/eps-community-strength.json'

with open(file_path, 'r') as file:
    data = json.load(file)

eps_records = []
modularity_values = []
intra_cluster_density_values = []

for eps, record in tqdm(data.items()):
    eps_records.append(float(eps))
    modularity_values.append(record['modularity'])
    intra_cluster_density_values.append(record['intra_cluster_density'])

eps_records = np.array(eps_records)
modularity_values = np.array(modularity_values)
intra_cluster_density_values = np.array(intra_cluster_density_values)

# 对 eps_records 排序并应用于其他数组
sorted_indices = np.argsort(eps_records)
eps_records = eps_records[sorted_indices]
modularity_values = modularity_values[sorted_indices]
intra_cluster_density_values = intra_cluster_density_values[sorted_indices]

# 创建平滑曲线
x_new = np.linspace(eps_records.min(), eps_records.max(), 500)

spl_modularity = make_interp_spline(eps_records, modularity_values, k=3)
modularity_smooth = spl_modularity(x_new)

spl_intra_cluster_density = make_interp_spline(eps_records, intra_cluster_density_values, k=3)
intra_cluster_density_smooth = spl_intra_cluster_density(x_new)

# 绘图
plt.figure(figsize=(16, 8))

# 绘制模块度图
plt.subplot(1, 2, 1)
plt.plot(x_new, modularity_smooth, color='blue', linewidth=2, label='Modularity')
plt.scatter(eps_records, modularity_values, color='blue', s=10)
plt.xlabel('Epsilon', fontsize=14)
plt.ylabel('Modularity', fontsize=14)
plt.title('Modularity vs. Epsilon', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# 绘制社区内部连接密度图
plt.subplot(1, 2, 2)
plt.plot(x_new, intra_cluster_density_smooth, color='red', linewidth=2, label='Intra-cluster Density')
plt.scatter(eps_records, intra_cluster_density_values, color='red', s=10)
plt.xlabel('Epsilon', fontsize=14)
plt.ylabel('Intra-cluster Density', fontsize=14)
plt.title('Intra-cluster Density vs. Epsilon', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("./imgs/eps_community_strength_subplot.png")
plt.show()
