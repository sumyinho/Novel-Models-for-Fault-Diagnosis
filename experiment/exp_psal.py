import torch
import pickle
import time
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from model.baseline_lstm import BaselineLSTM
from model.dsl import DeepResidualCNN
from model.psal import SparseAttentionLSTM
from util.dataset import VibrationSignalDataset
from util.utils import evaluate_model, generate_mask

# Variables and Super Parameters
random_seed = 3407
batch_size = 32
test_size = .2
learning_rate = 1e-3
epoches = 200
n_splits = 10
number_of_class = 3
channel_last = True  # if channel_last is true, the input dimension is (batch_size, sequence_length, channel_size).
torch.manual_seed(random_seed)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
root = "../dataset/XJTU-SY_Bearing_Datasets/"
# root = "../dataset/CWRU/"
# root = "../dataset/Paderborn_manual_damage/"

# with open(root + "PU.pkl", "rb") as f:
# with open(root + "CWRU.pkl", "rb") as f:
with open(root + "XJTU.pkl", "rb") as f:
    samples = pickle.load(f)
data = samples[:, :-1]
labels = samples[:, -1]
spliter = StratifiedShuffleSplit(n_splits=n_splits, random_state=random_seed, test_size=test_size)
k_flod_idx = []  # Storage index
# Storage measurement result about model. (accuracy, precision, recall, f1, conf_matrix, loss and running time)
k_storage_measurement_result = []
for index in spliter.split(data, labels):
    k_flod_idx.append(index)

for train_index, test_index in k_flod_idx:
    train_dataset = VibrationSignalDataset(data[train_index], labels[train_index], channel_last=channel_last)
    test_dataset = VibrationSignalDataset(data[test_index], labels[test_index], channel_last=channel_last)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    test_data = torch.FloatTensor(np.expand_dims(data[test_index], axis=1))
    test_labels = torch.LongTensor(labels[test_index])
    loss_list = []
    if channel_last is True:
        test_data = test_data.transpose(-1, -2)
    # Define model and its duplication
    model = SparseAttentionLSTM(sequence_length=1024, hidden_size=100, number_of_class=number_of_class)
    mask = generate_mask(1024, 32)
    model_best = model
    accuracy_best = 0
    model = model.to(device)
    mask = mask.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    start_time = time.time()  # Record start time
    for epoch in range(epoches):
        # Train Model
        model.train()
        for x, y in train_dataloader:
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x, mask)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        # Test Model
        with torch.no_grad():
            model.eval()
            correct_prediction_num = 0
            accuracy = 0
            for x, y in test_dataloader:
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x, mask)
                fault_mode = y_pred.max(1)[1]
                correct_prediction_num += torch.count_nonzero(y.eq(fault_mode)).item()
            accuracy = correct_prediction_num / len(test_dataset)
        print(f"epoch:{epoch}, loss:{loss}, accuracy:{accuracy}")
        loss_list.append(loss.item())
        if accuracy > accuracy_best:  # Restore model that have best accuracy
            accuracy_best = accuracy
            model_best = model
    end_time = time.time()
    time_taken = end_time - start_time  # Count time
    print("Time taken for the model to run: {:.2f} seconds".format(time_taken))
    # Evaluate model(including precision, recall, f1, roc_auc and conf_matrix)
    test_data = test_data.to(device)  # copy data to cuda
    test_labels = test_labels.to(device)
    precision, recall, f1, conf_matrix = \
        evaluate_model(model_best, test_data, test_labels)
    print(f"accuracy:{accuracy_best}, precision:{precision}, recall:{recall}, f1:{f1}")
    # Storage measurement result about model
    k_storage_measurement_result.append((accuracy_best, precision, recall, f1, conf_matrix,
                                         loss_list, time_taken, model_best))
with open("../experiment_data/dsl_XJTU.pkl", "wb") as f:
    pickle.dump(k_storage_measurement_result, f)