import torch
import torch.nn as nn
import numpy as np
import d2lzh_pytorch as d2l

# 最简单的深度网络称为多层感知机。多层感知机由多层神经元组成，每一层与它的上一层相连，从中接收输入；同时每一层也与它的下一层相连，影响当前层的神经元。

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, root='./data_mnist')
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype = torch.float)
b1 = torch.zeros(num_hiddens, dtype = torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype = torch.float)
b2 = torch.zeros(num_outputs, dtype = torch.float)
params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad = True)

# 激活函数
def relu(X):
    return torch.max(input = X, other = torch.tensor(0.0))

def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2

loss = nn.CrossEntropyLoss()
num_epochs, lr = 5, 100.0
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)
