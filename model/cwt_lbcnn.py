import torch
import torch.nn as nn
import torch.nn.functional as F


# Title: Intelligent fault diagnosis of rotating machinery based on continuous wavelet
# transform-local binary convolutional neural network
# DOI: 10.1016/j.knosys.2021.106796
class LocalBinaryCNN(nn.Module):
    def __init__(self, number_of_class=6, dropout=.1):
        super(LocalBinaryCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Local Binary Convolution
        self.conv3 = LocalBinaryConv2d(16, 32, kernel_size=5)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(103968, 1600),
            nn.Dropout(dropout),
            nn.Linear(1600, number_of_class)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print("x.shape:", x.shape)
        x = self.fc(x)
        return F.softmax(x, dim=-1)


class LocalBinaryConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(LocalBinaryConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_bin = torch.sign(x)
        x = x_bin * x_norm
        x = self.conv(x)
        return x
