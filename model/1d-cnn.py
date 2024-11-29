import torch.nn as nn


class OneDimensionCNN(nn.Module):

    def __init__(self):
        super(OneDimensionCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=23, bias=True)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=13, bias=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.linear = nn.Linear()

    # de-averaging operation
    def de_averaging(self, x):
        return x - x.mean()

    def forward(self, x):
        # Convolution Group 1
        x = self.conv1(x)
        x = self.de_averaging(x)
        x = self.relu(x)
        x = x / 23
        # Convolution Group 2
        x = self.conv2(x)
        x = self.de_averaging(x)
        x = self.relu(x)
        x = x / 12
        return x
