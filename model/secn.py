import torch
import torch.nn as nn
import torch.nn.functional as F


# Title: An Efficient Sequential Embedding ConvNet for Rotating Machinery Intelligent Fault Diagnosis
# DOI: 10.1109/TIM.2023.3267376
class SequentialEmbeddingConvNet(nn.Module):

    def __init__(self, number_of_class):
        super(SequentialEmbeddingConvNet, self).__init__()

        self.seq_embedding = nn.Conv1d(in_channels=1, out_channels=96, kernel_size=16, stride=16)
        self.conv1 = SeparableConvolution(in_channels=96, out_channels=96, kernel_size=3)
        self.conv2 = SeparableConvolution(in_channels=96, out_channels=96, kernel_size=9)
        self.conv3 = SeparableConvolution(in_channels=96, out_channels=96, kernel_size=15)

        self.global_pooling = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(288, number_of_class)
        )

    def forward(self, x):
        x = self.seq_embedding(x)
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_3 = self.conv3(x)
        x = torch.cat([x_1, x_2, x_3], dim=1)
        x = self.global_pooling(x)
        x = self.fc(x)
        return F.softmax(x, dim=-1)


class SeparableConvolution(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(SeparableConvolution, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.GELU(),
            nn.BatchNorm1d(out_channels)
        )
        self.pointwise_conv = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pointwise_conv(x)
        return x
