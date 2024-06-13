import os
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.optim as optim
from torch_geometric.data import Data
from sklearn.datasets import make_moons  
from torchmetrics import Accuracy
from tqdm.auto import tqdm
import json
from util import moon
import matplotlib.patheffects as PathEffects
from multiprocessing import Pool

np.random.seed(0)
torch.manual_seed(0)

n = 2000
m = 500

x, y = moon(n)
n_train = int(n * 0.7)
train_ind = torch.randperm(n)[:n_train]
test_ind = torch.LongTensor(list(set(np.arange(n)) - set(train_ind.tolist())))
D = pairwise_distances(x)
MAX_DISTANCE = np.max(D)

# Precompute a mask for distances greater than zero
mask = D > 0

c = x[:, 0].argsort().argsort()
fig, ax = plt.figure(figsize=(5, 3)), plt.gca()
ax.scatter(x[:, 0], x[:, 1], c=c, s=10, rasterized=True)
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor('#eeeeee')
txt = ax.text(0.05, 0.05, 'Ground Truth', color='k', fontsize=14, weight='bold', transform=ax.transAxes)
txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='#eeeeee')])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Max Distance:", MAX_DISTANCE)
# print("Edge_Num:", len(edges))

#%%
def analyze_network_structure(edges):
    G = nx.Graph()
    G.add_edges_from(edges)
    avg_degree = np.mean([d for n, d in G.degree()])
    num_components = nx.number_connected_components(G)
    component_sizes = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    return avg_degree, num_components, component_sizes

def process_eps(eps):
    fr, to = np.where(D < eps)
    edges = list(zip(fr, to)) 
    avg_degree, num_components, component_sizes = analyze_network_structure(edges)
    return eps, {
        'edge_num': len(edges),
        'average_degree': avg_degree,
        'num_components': num_components,
        'component_sizes': component_sizes
    }

if __name__ == "__main__":
    eps_range = np.arange(0.01, 0.2, 0.01)
    results = {}

    # Use multiprocessing to speed up the computation
    with Pool() as pool:
        for eps, result in tqdm(pool.imap_unordered(process_eps, eps_range), total=len(eps_range)):
            print(result['num_components'])
            if(result['num_components'] == 1):
                break
            results[eps] = result

    # Save results to a file
    with open('./analysis/eps-boom-component-record.json', 'w') as file:
        json.dump(results, file, indent=4)