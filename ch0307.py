import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import d2lzh_pytorch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, root='./data_mnist')

num_inputs, num_outputs = 784, 10

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

from collections import OrderedDict

net = nn.Sequential(OrderedDict([('flattenlayer', FlattenLayer()), ('linear', nn.Linear(num_inputs, num_outputs))]))
init.normal_(net.linear.weight, mean = 0, std = 0.01)
init.constant_(net.linear.bias, val = 0)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.1)
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, optimizer = optimizer)
