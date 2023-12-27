import torch
import torch.nn as nn
import numpy as np
import d2lzh_pytorch as d2l
from matplotlib import pyplot as plt

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
# d2l.plot(x.detach(), y.detach(), 'f(x) = relu(x)', 'x', 'relu(x)', title='ch0401', figsize=(8, 4))
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', 'f(x) = relu(x)')

# 使用ReLU的原因是，它求导表现得特别好：要么让参数消失，要么让参数通过。这使得优化表现得更好，并且ReLU减轻了困扰以往神经网络的梯度消失问题（稍后将详细介绍）
y.backward(torch.ones_like(x), retain_graph=True) # 自动求导
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))

y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))

# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))

y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))

# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))