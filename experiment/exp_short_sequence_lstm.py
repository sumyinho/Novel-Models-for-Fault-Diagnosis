import torch
import pickle

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from model.short_sequence_lstm import ShortSequenceLSTM
from util.dataset import VibrationSignalDataset
from util.preprocess import fold_signal

random_seed = 3407
batch_size = 16
test_size = .3
learning_rate = 0.0001
epoches = 200
torch.manual_seed(random_seed)
device = "cuda:0" if torch.cuda.is_available() else "cup"
root = "../dataset/XJTU-SY_Bearing_Datasets/"


with open(root + "XJTU.pkl", "rb") as f:
    samples = pickle.load(f)
data = samples[:, :-1]
labels = samples[:, -1]
spliter = StratifiedShuffleSplit(n_splits=1, random_state=random_seed, test_size=test_size)
k_flod_idx = []
for index in spliter.split(data, labels):
    k_flod_idx.append(index)

for train_index, test_index in k_flod_idx:
    train_dataset = VibrationSignalDataset(data[train_index], labels[train_index],
                                           transform=fold_signal, window_size=32)
    test_dataset = VibrationSignalDataset(data[test_index], labels[test_index],
                                          transform=fold_signal, window_size=32)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = ShortSequenceLSTM(sequence_length=32, input_dim=32, hidden_dim=200, dropout=.3)
    model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # schedule = MultiStepLR(optimizer, milestones=[40, 80], gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epoches):
        # Train Model
        model.train()
        for x, y in train_dataloader:
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
        # schedule.step()

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
                # print("y:", y)
                # print("fault_mode:", fault_mode)
                correct_prediction_num += torch.count_nonzero(y.eq(fault_mode)).item()
            accuracy = correct_prediction_num / len(test_dataset)
        print(f"epoch:{epoch}, loss:{loss}, accuracy:{accuracy}")

