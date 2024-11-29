import torch
import torch.nn as nn
import torch.nn.functional as F


# Title: Understanding and Learning Discriminant Features based on Multi-Attention 1DCNN for
# Wheelset Bearing Fault Diagnosis
# DOI: 10.1109/TII.2019.2955540
class Ma1dCNN(nn.Module):

    def __init__(self, number_of_class):
        super(Ma1dCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=32, padding='same'),
            nn.ReLU(),
            JointAttentionModule(in_channels=32)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=16, padding=7, stride=2),
            nn.ReLU(),
            JointAttentionModule(in_channels=32)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=9, padding=4, stride=2),
            nn.ReLU(),
            JointAttentionModule(in_channels=64)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=6, padding=2, stride=2),
            nn.ReLU(),
            JointAttentionModule(in_channels=64)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=4),
            nn.ReLU(),
            JointAttentionModule(in_channels=128)
        )
        self.conv6 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=4)
        self.global_avg_pooling = nn.Sequential(
            nn.AvgPool1d(kernel_size=8),
            nn.Flatten(),
            nn.Linear(128, number_of_class)
        )

    def forward(self, x):
        # print("input x:", x.shape)
        x = self.conv1(x)
        # print("conv1:", x.shape)
        x = self.conv2(x)
        # print("conv2:", x.shape)
        x = self.conv3(x)
        # print("conv3:", x.shape)
        x = self.conv4(x)
        # print("conv4:", x.shape)
        x = self.conv5(x)
        # print("conv5:", x.shape)
        x = self.conv6(x)
        # print("conv6:", x.shape)
        # x = self.linear(x)
        x = self.global_avg_pooling(x)
        # print("Average Pooling:", x.shape)
        # print("Linear: ", x.shape)
        return F.softmax(x, dim=-1)


# CAM
class ChannelAttentionModule(nn.Module):

    def __init__(self, in_channels):
        super(ChannelAttentionModule, self).__init__()
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(output_size=1)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels // 2, out_channels=in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    # input size:(batch_size, channels, features)
    def forward(self, x):
        residual = x
        # attention = x.mean(dim=-1)
        attention = self.global_avg_pooling(x)  # Global Average Pooling
        attention = self.conv1(attention)
        attention = self.conv2(attention)
        x = x * attention
        x = x + residual
        return F.relu(x)


# EAM
class ExcitationAttentionModule(nn.Module):

    def __init__(self, in_channels):
        super(ExcitationAttentionModule, self).__init__()
        self.in_channels = in_channels

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x):
        residual = x
        attention = self.conv1(x)
        # attention = attention.unsqueeze(dim=1)
        x = self.conv2(x)
        # x = x.reshape(-1, self.in_channels, self.sequence_length)
        x = x * attention
        x = x + residual
        return x


# JAM
class JointAttentionModule(nn.Module):

    def __init__(self, in_channels):
        super(JointAttentionModule, self).__init__()

        self.eam = ExcitationAttentionModule(in_channels=in_channels)
        self.cam = ChannelAttentionModule(in_channels=in_channels)

    def forward(self, x):
        x = self.eam(x)
        x = self.cam(x)
        return x
