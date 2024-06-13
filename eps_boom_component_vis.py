#%%
from matplotlib import pyplot as plt
import json
from tqdm import tqdm
import numpy as np

file_path = './analysis/eps-boom-component-record.json'
with open(file_path, 'r') as file:
    data = json.load(file)

eps_records = []
largest_components = []
second_components = []

for eps, record in tqdm(data.items()):
    eps_records.append(float(eps))  
    largest_components.append(record['component_sizes'][0])
    second_components.append(record['component_sizes'][1] if len(record['component_sizes']) > 1 else 0)

eps_records = np.array(eps_records)
largest_components = np.array(largest_components)
second_components = np.array(second_components)

sorted_indices = np.argsort(eps_records)
eps_records = eps_records[sorted_indices]
largest_components = largest_components[sorted_indices]
second_components = second_components[sorted_indices]

plt.figure(figsize=(12, 8))

plt.scatter(eps_records, largest_components, color='blue', s=10)
plt.scatter(eps_records, second_components, color='red', s=10)

from scipy.interpolate import make_interp_spline

x_new = np.linspace(eps_records.min(), eps_records.max(), 500) 

spl_largest = make_interp_spline(eps_records, largest_components, k=3)  # 使用 B 样条曲线，k=3 表示三次样条
largest_smooth = spl_largest(x_new)

spl_second = make_interp_spline(eps_records, second_components, k=3)
second_smooth = spl_second(x_new)

plt.plot(x_new, largest_smooth, label='Largest Component Size', color='blue', linewidth=2)
plt.plot(x_new, second_smooth, label='Second Largest Component Size', color='red', linewidth=2, linestyle='--')

plt.xlabel('Epsilon', fontsize=14)
plt.ylabel('Component Size', fontsize=14)
plt.title('Component Sizes vs. Epsilon', fontsize=16)

plt.legend(fontsize=12)

plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig("./imgs/eps_boom_component_record.png")
plt.show()
