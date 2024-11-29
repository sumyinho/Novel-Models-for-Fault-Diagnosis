import torch
import torch.nn as nn
import torch.nn.functional as F


# Title: Rolling Bearing Fault Diagnosis Method Base on Periodic Sparse Attention and LSTM
# DOI: 10.1109/JSEN.2022.3173446
class SparseAttentionLSTM(nn.Module):

    def __init__(self, sequence_length, hidden_size, number_of_class):
        super(SparseAttentionLSTM, self).__init__()
        self.attention = SingleHeadAttention(sequence_length)
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, number_of_class)

    def forward(self, x, mask):
        attention_weight = self.attention(x, mask)
        x = attention_weight.bmm(x)
        _, (h_n, _) = self.lstm(x)
        x = self.linear(h_n[0])
        return F.softmax(x, dim=-1)


class SingleHeadAttention(nn.Module):

    def __init__(self, sequence_length):
        super(SingleHeadAttention, self).__init__()
        self.sequence_length = torch.FloatTensor(sequence_length).cuda()

        self.linear_q = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sequence_length, sequence_length)
        )
        self.linear_k = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sequence_length, sequence_length)
        )

        self.linear_v = nn.Sequential(
            nn.Flatten(),
            nn.Linear(sequence_length, sequence_length)
        )

    def forward(self, x, mask):
        # q = self.linear_q(x)
        # k = self.linear_k(x)
        # v = self.linear_v(x)
        x = (x @ x.T)/self.sequence_length @ x
        # x = q @ k.T @ v
        x = x.unsqueeze(dim=-1)
        x = x * mask
        return F.softmax(x, dim=-1)


# def generate_mask(n, T):
#     """
#     @:param n: sequence length
#     @:param T: Window size
#     """
#     mask = torch.zeros(size=(n, n))
#     for i in range(n):
#         start = i - T
#         end = i + T + 1
#         if start < 0:
#             start = 0
#         if end > n:
#             end = n
#         mask[i][start:end] = 1
#         mask[i][i + 2 * T:n:T] = 1
#     return mask