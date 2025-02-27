import numpy as np

# 加载 .npy 文件
data = np.load('/users/ymy_yuan/m3/data/shard5_nflows100_nhosts3_lr10Gbps/feat_topo-pl-3_s0.npz')

for key in data.keys():
    print(f"Key: {key}, Shape: {data[key].shape}, Dtype: {data[key].dtype}")
