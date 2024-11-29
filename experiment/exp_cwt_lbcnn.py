# model = MultiScaleDeepResidualLSTM(in_channels=1, out_channels=10, kernel_size=10, hidden_size=600)
import torch
import pickle
import time
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from model.cwt_lbcnn import LocalBinaryCNN
from model.mdrl_slstm import MultiScaleDeepResidualLSTM
from util.dataset import VibrationSignalDataset, TimeFrequencyDataset
from util.utils import evaluate_model

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Variables and Super Parameters
random_seed = 3407
batch_size = 32
test_size = .2
learning_rate = 1e-3
epoches = 200
n_splits = 10
number_of_class = 6
channel_last = False  # if channel_last is true, the input dimension is (batch_size, sequence_length, channel_size).
torch.manual_seed(random_seed)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# root = "../dataset/XJTU-SY_Bearing_Datasets/"
# root = "../dataset/CWRU/"
# root = "../dataset/Paderborn_manual_damage/"

# root_dict = {"../dataset/CWRU/": "CWRU", "../dataset/Paderborn_manual_damage/": "PU",
#              "../dataset/XJTU-SY_Bearing_Datasets/": "XJTU"}
root_dict = {"../dataset/XJTU-SY_Bearing_Datasets/": "XJTU-cwt"}
# with open(root + "PU.pkl", "rb") as f:
# with open(root + "CWRU.pkl", "rb") as f:
for root, dataset_name in root_dict.items():
    with open(root + dataset_name + ".pkl", "rb") as f:
        samples = pickle.load(f)
    if dataset_name == "XJTU-cwt":
        number_of_class = 3
    data = samples['data']
    labels = samples['label']
    spliter = StratifiedShuffleSplit(n_splits=n_splits, random_state=random_seed, test_size=test_size)
    k_flod_idx = []  # Storage index
    # Storage measurement result about model. (accuracy, precision, recall, f1, conf_matrix, loss and running time)
    k_storage_measurement_result = []
    for index in spliter.split(data, labels):
        k_flod_idx.append(index)

    for train_index, test_index in k_flod_idx:
        # train_dataset = VibrationSignalDataset(data[train_index], labels[train_index], channel_last=channel_last)
        # test_dataset = VibrationSignalDataset(data[test_index], labels[test_index], channel_last=channel_last)
        train_dataset = TimeFrequencyDataset(data[train_index], labels[train_index])
        test_dataset = TimeFrequencyDataset(data[test_index], labels[test_index])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        test_data = torch.FloatTensor(np.expand_dims(data[test_index], axis=1))
        test_labels = torch.tensor(labels[test_index], dtype=torch.long)
        loss_list = []
        if channel_last is True:
            test_data = test_data.transpose(-1, -2)
        # Define model and its duplication
        # model = MultiScaleDeepResidualLSTM(hidden_size=100, number_of_class=number_of_class)
        model = LocalBinaryCNN(number_of_class=number_of_class)
        model = model.to(device)
        model_best = model
        accuracy_best = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        start_time = time.time()  # Record start time
        for epoch in range(epoches):
            # Train Model
            model.train()
            running_loss = 0
            for x, y in train_dataloader:
                optimizer.zero_grad()
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            # Test Model
            with torch.no_grad():
                model.eval()
                correct_prediction_num = 0
                accuracy = 0
                for x, y in test_dataloader:
                    x = x.to(device)
                    y = y.to(device)
                    y_pred = model(x)
                    fault_mode = y_pred.max(1)[1]
                    # print("fault_mode:", fault_mode)
                    correct_prediction_num += torch.count_nonzero(y.eq(fault_mode)).item()
                accuracy = correct_prediction_num / len(test_dataset)
            print(f"epoch:{epoch}, training loss:{running_loss/len(train_dataset)}, test accuracy:{accuracy}")
            loss_list.append(running_loss/len(train_dataset))
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
    model_name = "mdrl_slstm_"
    with open("../experiment_data/"+model_name+dataset_name+".pkl", "wb") as f:
        pickle.dump(k_storage_measurement_result, f)
    torch.cuda.empty_cache()  # 清空缓存




