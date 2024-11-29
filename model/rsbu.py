import torch
import torch.nn as nn
import torch.nn.functional as F


# Title: Deep Residual Shrinkage Networks for Fault Diagnosis
# DOI: 10.1109/tii.2019.2943898
class DeepResidualShrinkageNetwork(nn.Module):  # abbr. DRSN

    def __init__(self, number_of_class):
        super(DeepResidualShrinkageNetwork, self).__init__()

        self.conv = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1)
        self.rsbu1 = ResidualShrinkageBuildingUnit(in_channels=4, out_channels=4, kernel_size=3,
                                                   sequence_length=512, stride=2)
        self.rsbu2 = nn.ModuleList([
            ResidualShrinkageBuildingUnit(in_channels=4, out_channels=4, kernel_size=3,
                                          sequence_length=256, stride=1)
            for i in range(3)
        ])
        self.rsbu3 = ResidualShrinkageBuildingUnit(in_channels=4, out_channels=8, kernel_size=3,
                                                   sequence_length=256, stride=2)
        self.rsbu4 = nn.ModuleList([
            ResidualShrinkageBuildingUnit(in_channels=8, out_channels=8, kernel_size=3,
                                          sequence_length=128, stride=1)
            for i in range(3)
        ])
        self.rsbu5 = ResidualShrinkageBuildingUnit(in_channels=8, out_channels=16, kernel_size=3,
                                                   sequence_length=128, stride=2)
        self.rsbu6 = nn.ModuleList([
            ResidualShrinkageBuildingUnit(in_channels=16, out_channels=16, kernel_size=3,
                                          sequence_length=64, stride=1)
            for i in range(3)
        ])
        self.BN = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, number_of_class)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.rsbu1(x)
        for layer in self.rsbu2:
            x = layer(x)
        # x = self.rsbu2(x)
        x = self.rsbu3(x)
        # x = self.rsbu4(x)
        for layer in self.rsbu4:
            x = layer(x)
        x = self.rsbu5(x)
        # x = self.rsbu6(x)
        for layer in self.rsbu6:
            x = layer(x)
        x = self.BN(x)
        x = self.relu(x)
        # print("Before Pooling:", x.shape)
        x = x.mean(dim=-1)
        # print("After Pooling:", x.shape)
        x = self.fc(x)
        return F.softmax(x, dim=1)


class ResidualShrinkageBuildingUnit(nn.Module):  # abbr. RSBU

    def __init__(self, in_channels, out_channels, kernel_size, sequence_length,
                 stride=1, padding=1):
        super(ResidualShrinkageBuildingUnit, self).__init__()

        self.device = "cuda:0" if torch.cuda.is_available() else "cup"
        self.stride = stride
        self.out_channels = out_channels
        self.sequence_length = sequence_length
        if self.stride > 1:
            self.shortcut = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_channels * sequence_length,
                          out_channels * sequence_length // 2),
            )
        self.conv1 = nn.Sequential(  # convolutional Layer 1
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                      padding=padding, stride=stride),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(  # convolutional Layer 2
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                      padding=padding, stride=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        self.linear = nn.Sequential(  # Thresholds
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if self.stride > 1:
            residual = self.shortcut(x)
            residual = residual.reshape(-1, self.out_channels, self.sequence_length // 2)
        else:
            residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        threshold = x.abs()
        threshold = threshold.mean(dim=-1)  # Global Average Pooling
        a = self.linear(threshold)
        threshold = threshold.mean(dim=1, keepdim=True)
        threshold = threshold * a
        threshold = threshold.unsqueeze(dim=-1)
        threshold = threshold.repeat(1, x.shape[1], x.shape[2])
        # print("threshold:", threshold.shape)  # Test Code
        # print("x:", x.shape)
        zeros = torch.zeros(size=x.shape, device=self.device)
        x = torch.where(x.abs() <= threshold, zeros, x)  # Judge every elements in x.
        x = torch.where(x > threshold, x - threshold, x)
        x = torch.where(x < (-threshold), x + threshold, x)
        # print("After where:", x.shape)
        x = x + residual
        # print("x + residual:", x.shape)
        return x
