import torch
import torch.nn as nn


class BaselineCNN(nn.Module):

    def __init__(self, number_of_class, dropout=.2):
        super(BaselineCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=23),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=12),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3920, 32),
            nn.Dropout(dropout),
            nn.Linear(32, number_of_class)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return torch.softmax(x, dim=1)
