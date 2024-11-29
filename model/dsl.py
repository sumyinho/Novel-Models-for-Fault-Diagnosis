import torch.nn as nn
import torch.nn.functional as F


# Title: Deep residual learning-based fault diagnosis method for rotating machinery
# DOI: 10.1016/j.isatra.2018.12.025
class DeepResidualCNN(nn.Module):

    def __init__(self, number_of_class):
        super(DeepResidualCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=10, kernel_size=10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
        )

        self.residual_block1 = BuildingBlock()
        self.pooling = nn.MaxPool1d(2)
        self.residual_block2 = BuildingBlock()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(5070, number_of_class),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.residual_block1(x)
        x = self.pooling(x)
        x = self.residual_block2(x)
        # print("x.shape:", x.shape)
        x = self.fc(x)
        return F.softmax(x, dim=1)


class BuildingBlock(nn.Module):

    def __init__(self):
        super(BuildingBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=10, kernel_size=10, padding='same'),
            nn.BatchNorm1d(10),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=10, kernel_size=10, padding='same'),
            nn.BatchNorm1d(10),
            nn.ReLU(),
        )

    def forward(self, x):
        residual = x
        # print("input shape:", residual.shape)
        x = self.conv1(x)
        # print("Conv1d_1:", x.shape)
        x = self.conv2(x)
        # print("Conv1d_2:", x.shape)
        x = x + residual
        x = F.relu(x)
        return x
