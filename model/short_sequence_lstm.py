import torch.nn as nn
import torch.nn.functional as F


class ShortSequenceLSTM(nn.Module):

    def __init__(self, sequence_length, input_dim, hidden_dim, dropout=.1):
        super(ShortSequenceLSTM, self).__init__()

        self.input_linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

        self.linear = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(sequence_length * input_dim, 3),
        )

    # input size: (batch_size, sequence_length, feature_size)
    def forward(self, x):
        x = self.input_linear(x)
        h_n, _ = self.lstm(x)
        x = self.linear(h_n)
        return F.softmax(x, dim=-1)
