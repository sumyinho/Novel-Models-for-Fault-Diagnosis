import torch
import torch.nn as nn
import torch.nn.functional as F


# Title: Interpreting Network Knowledge with Attention Mechanism for Bearing Fault Diagnosis
# DOI: 10.1016/j.asoc.2020.106829
class AttMBiGRU(nn.Module):

    def __init__(self, sequence_length, number_of_class, dropout=0):
        super(AttMBiGRU, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(kernel_size=1, in_channels=64, out_channels=64, padding='same'),
            nn.Conv1d(kernel_size=3, in_channels=64, out_channels=128, padding='same'),
            nn.LeakyReLU(0.1),
        )
        self.gru = nn.GRU(input_size=128, hidden_size=128, num_layers=2, bidirectional=True,
                          batch_first=True, dropout=dropout)
        self.att = AttentionLayer(channel_size=128, dropout=dropout)
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * sequence_length, number_of_class)
        )

    def forward(self, x):
        # input size: (batch_size, channel_size, sequence_length)
        x = self.cnn(x)
        x = x.transpose(-1, -2)
        # input size: (batch_size, sequence_length, feature_size)
        x, _ = self.gru(x)
        x_tmp = x.squeeze(dim=0).reshape(2, -1, 16, 128)
        pos, neg = x_tmp[0].squeeze(), x_tmp[1].squeeze()
        x_prime = pos + neg
        x = self.att(x, x_prime)
        # print("x.shape:", x.shape)
        x = self.linear(x)
        return F.softmax(x, dim=-1)


class AttentionLayer(nn.Module):

    def __init__(self, channel_size, dropout):
        super(AttentionLayer, self).__init__()

        self.linear = nn.Linear(channel_size, channel_size)
        self.dropout = nn.Dropout(dropout)

    def score_function(self, x_prime):
        q = self.linear(x_prime)
        score = torch.bmm(torch.tanh(x_prime), q.transpose(-1, -2))
        alpha = F.softmax(score, dim=-1)
        return alpha

    def forward(self, x, x_prime):
        # input size: (batch_size, sequence_length, feature_size)
        # print("x_att:", x.shape)
        # print("x_att_prime:", x_prime.shape)
        alpha = self.score_function(x_prime)
        y_att = torch.bmm(alpha, x)
        y_att = torch.tanh(y_att)
        # print("y_att:", y_att.shape)
        return self.dropout(y_att)
