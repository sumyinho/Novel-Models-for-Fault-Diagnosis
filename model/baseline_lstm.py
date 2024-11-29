import torch
import torch.nn as nn


# 定义模型
# 【注】：出现了一次模式崩溃，后续改进应该增加模型的非线性。
class BaselineLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=.2, sequence_length=1024,
                 batch_first=True):
        super(BaselineLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=batch_first)
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size*sequence_length, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # print("x.shape:", x.shape)
        out, _ = self.lstm(x, None)
        # print("out.shape:", out.shape)
        x = self.linear(out)
        return torch.softmax(x, dim=1)
