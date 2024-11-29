import torch
import torch.nn as nn
import torch.nn.functional as F


# Title: Gearbox fault diagnosis based on Multi-Scale deep residual learning and stacked LSTM model
# DOI: 10.1016/j.measurement.2021.110099
class MultiScaleDeepResidualLSTM(nn.Module):

    def __init__(self, hidden_size, number_of_class, dropout_rate=.2):
        super(MultiScaleDeepResidualLSTM, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=10, kernel_size=10, padding='same'),
            nn.MaxPool1d(2),
            ResidualBlock(in_channels=10, out_channels=10, kernel_size=10),
            ResidualBlock(in_channels=10, out_channels=10, kernel_size=10),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=10, kernel_size=10, padding='same'),
            nn.MaxPool1d(2),
            ResidualBlock(in_channels=10, out_channels=10, kernel_size=10),
            ResidualBlock(in_channels=10, out_channels=10, kernel_size=10),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),
        )

        self.lstm = nn.LSTM(input_size=10, hidden_size=hidden_size, num_layers=3, batch_first=True)
        # 该部分需要修改
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(100, number_of_class),
        )

    def forward(self, x):
        x1 = self.cnn1(x)
        x2 = self.cnn2(x)

        x = torch.cat((x1, x2), dim=-1)  # 拼接在一起
        # print("concat x:", x.shape)
        x = x.transpose(-1, -2)  # 交换两个维度
        # print("after transpose x:", x.shape)
        x, _ = self.lstm(x)
        # print("after lstm x[:, -1, :]:", x[:, -1, :].shape)
        x = self.linear(x[:, -1, :])
        # print("before softmax:", x.shape)
        return F.softmax(x, dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        return F.leaky_relu(x)
