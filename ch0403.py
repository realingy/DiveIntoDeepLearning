import torch
from torch import nn
from torch.nn import init
import numpy as np
import d2lzh_pytorch as d2l

num_inputs, num_outputs, num_hiddens = 784, 10, 256
net = nn.Sequential(
                d2l.FlattenLayer(), 
                nn.Linear(num_inputs, num_hiddens), 
                nn.ReLU(),
                nn.Linear(num_hiddens, num_outputs), 
            )

for params in net.parameters():
    init.normal_(params, mean = 0, std = 0.01)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, root='./data_mnist')
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = 0.5)
num_epoches = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epoches, batch_size, None, None, optimizer)