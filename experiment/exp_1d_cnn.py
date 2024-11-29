import torch
import numpy as np
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from util.dataset import VibrationSignalDataset


random_seed = 3407
batch_size = 32
test_size = .3
learning_rate = 1e-3
epoches = 200
device = "cuda:0" if torch.cuda.is_available() else "cpu"
root = "../dataset/XJTU-SY_Bearing_Datasets/"
# step 1: Load and split dataset
with open(root + "XJTU.pkl", "rb") as f:
    samples = pickle.load(f)
data = samples[:, :-1]
labels = samples[:, -1]
spliter = StratifiedShuffleSplit(n_splits=10, random_state=random_seed, test_size=test_size)
k_flod_idx = []
for index in spliter.split(data, labels):
    k_flod_idx.append(index)

for train_index, test_index in k_flod_idx:
    train_dataset = VibrationSignalDataset(data[train_index], labels[train_index])
    test_dataset = VibrationSignalDataset(data[test_index], labels[test_index])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # 实例化模型
    model = None # 假设模型不需要额外的初始化参数
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameter(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epoches):
        # 训练模型
        # 测试模型
        pass