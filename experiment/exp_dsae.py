import torch
import pickle

import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from model.dsae import DSAE, DSAE_Discriminator
from util.dataset import VibrationSignalDataset

# hyperparameter setting
random_seed = 3407
batch_size = 32
test_size = .3
learning_rate = 1e-3
epoches = 200
torch.manual_seed(random_seed)
torch.autograd.set_detect_anomaly(True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
root = "../dataset/XJTU-SY_Bearing_Datasets/"

# Load data and create dataset
with open(root + "XJTU.pkl", "rb") as f:
    samples = pickle.load(f)
data = samples[:, :-1]
labels = samples[:, -1].astype('int')
# one-hot class label
length = len(labels)
labels_one_hot = np.zeros(shape=(length, 3))
labels_one_hot[np.arange(length), labels] = 1
spliter = StratifiedShuffleSplit(n_splits=1, random_state=random_seed, test_size=test_size)
k_flod_idx = []
for index in spliter.split(data, labels_one_hot):
    k_flod_idx.append(index)

# Training models
for train_index, test_index in k_flod_idx:
    train_dataset = VibrationSignalDataset(data[train_index], labels_one_hot[train_index])
    test_dataset = VibrationSignalDataset(data[test_index], labels_one_hot[test_index])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # Create and setting model
    model = DSAE(in_units=1024, hidden_units=128)
    discriminator = DSAE_Discriminator(in_units=128, out_units=3)
    model.to(device)
    discriminator.to(device)
    optimizer = torch.optim.LBFGS(list(model.parameters()) + list(discriminator.parameters()),
                                  lr=learning_rate)
    # Setting Loss parameters
    criterion_DR = torch.nn.BCELoss()
    criterion_AE = torch.nn.MSELoss()
    criterion_KL = torch.nn.KLDivLoss()
    # criterion = torch.nn.CrossEntropyLoss()
    alpha = 0.1
    beta = 0.1

    for epoch in range(epoches):
        # Train Model
        model.train()
        for x, y in train_dataloader:
            optimizer.zero_grad()
            x = x.to(device)
            y = y.float().to(device)
            x_re, hidden = model(x)
            y_pred = discriminator(hidden)
            L_DR = criterion_DR(y_pred, y)
            L_AE = criterion_AE(x_re, x)
            L_KL = model.get_sparse_loss(hidden)
            L_DSAE = L_AE + L_DR + L_KL
            L_DSAE.backward()
            optimizer.step(closure=lambda: L_DSAE)

        # Test Model
        with torch.no_grad():
            model.eval()
            correct_prediction_num = 0
            accuracy = 0
            for x, y in test_dataloader:
                x = x.to(device)
                y = torch.argmax(y.to(device))
                _, hidden = model(x)
                y_pred = discriminator(hidden)
                fault_mode = y_pred.max(1)[1]
                correct_prediction_num += torch.count_nonzero(y.eq(fault_mode)).item()
            accuracy = correct_prediction_num / len(test_dataset)
        print(f"epoch:{epoch}, L_AE:{L_AE}, L_DR:{L_DR}, accuracy:{accuracy}")

