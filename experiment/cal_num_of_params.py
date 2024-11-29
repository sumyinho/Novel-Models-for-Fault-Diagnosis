import torchsummary
import torch

from model.baseline_cnn import BaselineCNN
from model.baseline_lstm import BaselineLSTM
from model.mixcnn import MIXCNN
from model.secn import SequentialEmbeddingConvNet

number_of_class = 6
# model = SequentialEmbeddingConvNet(number_of_class=number_of_class)
# model = MIXCNN(number_of_class=number_of_class)
# model = BaselineCNN(number_of_class=number_of_class)
model = BaselineLSTM(input_size=1, hidden_size=100, output_size=number_of_class,
                     num_layers=3, sequence_length=1024)
model.to("cuda:0")
x = torch.randn(size=(32, 1024, 1)).cuda()
out = model(x)
print(out.shape)
torchsummary.summary(model, (1024, 1))