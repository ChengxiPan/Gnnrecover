#%%
import json
import matplotlib.pyplot as plt

# 读取JSONL文件
file_path = './analysis/eps-gnnrecover_loss.jsonl'
epsilon = []
loss = []

with open(file_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        for key, value in data.items():
            epsilon.append(float(key))
            loss.append(value)

# 绘制图表
plt.figure(figsize=(10, 6))

epsilon = epsilon[: 30]
loss = loss[ :30]

plt.plot(epsilon, loss, marker='o', linestyle='-')
plt.grid(True)
plt.savefig("./imgs/eps_gnnrecover_dG3.png")
plt.show()

# %%
