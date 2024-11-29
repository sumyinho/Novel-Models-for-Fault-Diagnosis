import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


# Title: Selective kernel convolution deep residual network based on channel-spatial attention mechanism and feature
# fusion for mechanical fault diagnosis
# DOI: 10.1016/j.isatra.2022.06.035
class ChannelSpatialAttentionMechanism(nn.Module):

    def __init__(self, channel_size, r=2):
        super(ChannelSpatialAttentionMechanism, self).__init__()
        hidden_size = channel_size // r

        self.channel_max_pooling = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.channel_avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channel_size, hidden_size),
            nn.Linear(hidden_size, channel_size)
        )
        self.channel_sigmoid = nn.Sigmoid()  # Channel Attention

        self.spatial_max_pooling = nn.AdaptiveMaxPool2d(output_size=(None, 1))
        self.spatial_avg_pooling = nn.AdaptiveAvgPool2d(output_size=(None, 1))
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding="same")
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        # print("x:", x.shape)
        x_max = self.channel_max_pooling(x)
        x_avg = self.channel_avg_pooling(x)
        # print("x_max:", x_max.shape)
        # print("x_avg:", x_avg.shape)
        x_max = self.mlp(x_max)
        x_avg = self.mlp(x_avg)
        channel_attention = self.channel_sigmoid(x_max + x_avg)
        channel_attention = channel_attention.unsqueeze(dim=-1).unsqueeze(dim=-1)
        # print("The output of channel Attention:", channel_attention.shape)

        # Spatial Attention
        x_prime = x * channel_attention
        x_transpose = x_prime.transpose(1, -1)  # exchange channel and the last dimension.
        x_prime_max = self.spatial_max_pooling(x_transpose)
        x_prime_avg = self.spatial_avg_pooling(x_transpose)
        x_prime_max = x_prime_max.transpose(1, -1)
        x_prime_avg = x_prime_avg.transpose(1, -1)
        # print("x_prime_max:", x_prime_max.shape)
        # print("x_prime_avg:", x_prime_avg.shape)
        x_prime_concat = torch.cat([x_prime_max, x_prime_avg], dim=1)
        x_prime_concat = self.conv(x_prime_concat)
        # print("x_prime_concat:", x_prime_concat.shape)
        spatial_attention = self.spatial_sigmoid(x_prime_concat)
        return x_prime_concat * spatial_attention + x_prime


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ResidualBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                      padding=1),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()

        # self.csam = ChannelSpatialAttentionMechanism(channel_size=out_channels, stride=stride, padding=padding)
        self.csam = ChannelSpatialAttentionMechanism(channel_size=out_channels)

    def forward(self, x):
        # print("x.shape:", x.shape)
        residual = self.shortcut(x)
        x = self.conv(x)
        # print("after convolution:", x.shape)
        x = self.csam(x)
        # print("after csam:", x.shape)
        return F.relu(x + residual)


# Selective Kernel Convolution based on CSAM
class SelectiveKernelConvolutionCSAM(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, padding_1=0, padding_2=2, r=2):
        super(SelectiveKernelConvolutionCSAM, self).__init__()
        d = out_channels // r

        self.A = nn.Parameter(torch.empty(1, out_channels, d))
        self.B = nn.Parameter(torch.empty(1, out_channels, d))

        self.conv_u = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                      padding=padding_1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv_v = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=stride,
                      padding=padding_2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.csam_u = ChannelSpatialAttentionMechanism(channel_size=out_channels)
        self.csam_v = ChannelSpatialAttentionMechanism(channel_size=out_channels)
        # fuse:
        self.generate_z = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.Linear(out_channels, d)
        )
        self.fuse_bn_rl = nn.Sequential(
            nn.BatchNorm2d(d),
            nn.ReLU()
        )
        # selection:

        # Initial Parameters:
        self.__init_parameters()

    def __init_parameters(self):
        init.normal_(self.A)
        init.normal_(self.B)

    def forward(self, x):
        u = self.conv_u(x)
        v = self.conv_v(x)
        u_c = self.csam_u(u)
        v_c = self.csam_v(v)
        # print("u:", u.shape)
        # print("v:", v.shape)
        t = u + v
        # print("t.shape:", t.shape)
        z = self.generate_z(t)
        z = z.unsqueeze(-1).unsqueeze(-1)  # (batch_size, channels, H, W)
        z = self.fuse_bn_rl(z)
        batch_size = z.shape[0]  # get the batch_size
        # print("z.shape:", z.shape)
        # print("A.repeat:", self.A.repeat(batch_size, 1, 1).shape)
        A = torch.bmm(self.A.repeat(batch_size, 1, 1), z.squeeze(dim=-1))
        B = torch.bmm(self.B.repeat(batch_size, 1, 1), z.squeeze(dim=-1))
        score = torch.softmax(torch.cat([A, B], dim=2), dim=1)
        # print("score.shape:", score[:, :, 0].shape)
        score_u = score[:, :, 0].unsqueeze(dim=-1).unsqueeze(dim=-1)
        score_v = score[:, :, 1].unsqueeze(dim=-1).unsqueeze(dim=-1)
        # print("u_c.shape:", u_c.shape)
        # print("score_u.shape:", score_u.shape)
        Y = u_c * score_u + v_c * score_v
        # print("Y", Y.shape)
        return Y


class ResNetSKCS(nn.Module):

    def __init__(self):
        super(ResNetSKCS, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.conv3 = ResidualBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.conv4 = ResidualBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.conv5 = ResidualBlock(in_channels=256, out_channels=512, kernel_size=3, stride=2)

        # __init__(self, in_channels, out_channels, stride=1, padding_1=0, padding_2=2, r=2)
        self.skc_csam1 = SelectiveKernelConvolutionCSAM(in_channels=128, out_channels=512, stride=2,
                                                        padding_1=1, padding_2=3)
        self.skc_csam2 = SelectiveKernelConvolutionCSAM(in_channels=256, out_channels=512, stride=2,
                                                        padding_1=1, padding_2=3)
        self.skc_csam3 = SelectiveKernelConvolutionCSAM(in_channels=512, out_channels=512, stride=2,
                                                        padding_1=1, padding_2=3)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        x = self.conv1(x)
        # print("input_conv:", x.shape)
        x = self.conv2(x)
        low_level = self.conv3(x)
        middle_level = self.conv4(low_level)
        high_level = self.conv5(middle_level)
        low_level = self.skc_csam1(low_level)
        middle_level = self.skc_csam2(middle_level)
        high_level = self.skc_csam3(high_level)
        print("low_level:", low_level.shape)
        print("middle_level:", middle_level.shape)
        print("high_level:", high_level.shape)
        # upsampling
        middle_level = F.interpolate(middle_level, scale_factor=2, mode='bilinear', align_corners=True)
        high_level = F.interpolate(high_level, scale_factor=2, mode='bilinear', align_corners=True)
        feature1 = self.conv6(low_level)
        feature2 = self.conv6(middle_level)
        feature3 = self.conv6(high_level)
        print("feature1:", feature1.shape)
        print("feature2:", feature2.shape)
        print("feature3:", feature3.shape)
        # feature_cat = torch.cat([feature1, feature2, feature3], dim=1)
        feature_cat = feature1 + feature2 + feature3
        feature_cat = self.pooling(feature_cat)
        output = self.fc(feature_cat)
        return torch.softmax(output, dim=1)
