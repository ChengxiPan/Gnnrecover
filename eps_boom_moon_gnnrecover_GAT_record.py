#%%
from util import moon
from matplotlib import pyplot as plt
import matplotlib.patheffects as PathEffects
import torch
import numpy as np
from sklearn.metrics import pairwise_distances
import torch.optim as optim
from torch_geometric.data import Data
from tqdm.auto import tqdm
from scipy.linalg import orthogonal_procrustes
from util import Net, GIN, GAT, moon, stationary, reconstruct, dG
import argparse
import json

#%%
# def parse_args():
#     parser = argparse.ArgumentParser(description="Process some parameters.")
#     parser.add_argument('--epsilon', type=float, required=True, help='Epsilon value for the algorithm.')
#     args = parser.parse_args()
#     return args

def get_GAT_loss(epsilon):
    n = 2000
    m = 0

    x,y = moon(n)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_train = int(n * 0.7)
    train_ind = torch.randperm(n)[:n_train]
    test_ind = torch.LongTensor(list(set(np.arange(n)) - set(train_ind.tolist())))
    D = pairwise_distances(x)
    fr, to = np.where(D < epsilon)
    edges = list(zip(fr, to))

    # edge_index = np.vstack([fr, to])
    # edge_index = torch.tensor(edges, dtype=torch.long)
    x = torch.tensor(x, dtype=torch.float).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)
    edges = torch.tensor(edges, dtype=torch.long).t().contiguous()

    #%%~
    # Initialize model and optimizer
    net = GAT(m).to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()

    # Training loop
    # pbar = tqdm(range(100))

    for epoch in range(100):
        # ind = torch.eye(n)[:, torch.randperm(n)[:m]]
        # X_extended = torch.hstack([x, ind])  # Convert X to tensor
        data = Data(x=x, edge_index=edges).to(device)
        rec = net(data)  # reconstruct
        loss = dG(x[train_ind], rec[train_ind])  # train loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # description = f'Epsilon {epsilon}, Loss: {float(loss)}'
    # pbar.set_description(description)
    
    # record
    _data = {epsilon: float(loss)}
    output_file = './analysis/eps-gnnrecover_loss.jsonl'
    with open(output_file, 'a') as f:
        f.write(json.dumps(_data) + '\n')

    return loss
#%%
if __name__ == "__main__":
    losses = []
    eps_range = np.arange(0.01, 3.5, 0.02)
    
    pbar = tqdm(eps_range)
    for eps in pbar:
        loss = get_GAT_loss(eps)
        losses.append(loss)
        description = f'Epsilon {eps}, Loss: {float(loss)}'
        pbar.set_description(description)    
    
    plt.plot(losses)
    plt.grid(True)
    plt.savefig("./imgs/eps-gnnrecover_loss.png")
# %%
