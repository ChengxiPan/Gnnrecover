import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import json
from util import moon 
from multiprocessing import Pool
import torch
import matplotlib.patheffects as PathEffects


np.random.seed(0)

n = 2000
x, y = moon(n)
D = pairwise_distances(x)
MAX_DISTANCE = np.max(D)

c = x[:, 0].argsort().argsort()
fig, ax = plt.figure(figsize=(5, 3)), plt.gca()
ax.scatter(x[:, 0], x[:, 1], c=c, s=10, rasterized=True)
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor('#eeeeee')
txt = ax.text(0.05, 0.05, 'Ground Truth', color='k', fontsize=14, weight='bold', transform=ax.transAxes)
txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='#eeeeee')])

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Max Distance:", MAX_DISTANCE)

# 处理每个 eps 值的函数
def process_eps(eps):
    fr, to = np.where(D < eps)
    edges = list(zip(fr, to))
    
    # 构建图并计算社区结构
    G = nx.Graph()
    G.add_edges_from(edges)
    
    # 计算模块度
    partition = nx.community.greedy_modularity_communities(G)
    modularity = nx.community.modularity(G, partition)
    
    # 计算社区内部连接密度
    intra_cluster_density = sum([nx.density(G.subgraph(community)) for community in partition])
    
    return eps, {
        'modularity': modularity,
        'intra_cluster_density': intra_cluster_density
    }

if __name__ == "__main__":
    eps_range = np.arange(0.01, 0.2, 0.01)
    results = {}

    # 使用多进程加速计算
    with Pool() as pool:
        for eps, result in tqdm(pool.imap_unordered(process_eps, eps_range), total=len(eps_range)):
            results[eps] = result

    # 保存结果到文件
    with open('./analysis/eps-community-strength.json', 'w') as file:
        json.dump(results, file, indent=4)
