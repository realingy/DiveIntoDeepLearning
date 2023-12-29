import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import d2lzh_pytorch as d2l
import os # Python的标准库，用于与操作系统交互

# 3.12

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 设置训练样本数（n_train）、测试样本数（n_test）和输入特征数（num_inputs）。
n_train, n_test, num_inputs = 20, 100, 200
# 定义了真实的权重（true_w）和偏置（true_b），权重被设置为0.01，偏置被设置为0.05
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05
# 创建了一个随机数张量，这些随机数来自均值为0、标准差为1的正态分布。张量的形状为样本总数（n_train + n_test）乘以输入特征数（num_inputs）
features = torch.randn((n_train + n_test, num_inputs))
# 标签，使用特征和真实权重的矩阵乘法，然后加上真实的偏置。
labels = torch.matmul(features, true_w) + true_b
# 加入了高斯噪声，均值为0，标准差为0.01，模拟现实数据
labels += torch.tensor(np.random.normal(0, 0.01, size = labels.size()), dtype = torch.float)
# 将特征和标签分为训练集和测试集
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]

# 始化模型的参数:权重向量和偏置项。权重随机初始化，偏置初始化为零。requires_grad = True表示在优化过程中需要计算这些张量的梯度。
def init_params():
    w = torch.randn((num_inputs, 1), requires_grad = True)
    b = torch.zeros(1, requires_grad = True)
    return [w, b]

# 定义L2正则化项。L2正则化是一种常见的正则化技术，可以减少模型的过拟合。这个函数在训练过程中会被用来计算权重的L2范数（即欧几里得范数），作为正则化损失添加到总损失中。
def l2_penalty(w):
    return (w ** 2).sum() / 2

# 超参：batch_size 表示每批训练数据的大小，num_epochs 是训练轮数，lr 是学习率。
batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = d2l.linreg, d2l.squared_loss
dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle = True)

# 这个函数接受一个参数 lambd，它是L2正则化项的权重。
def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            # 计算损失，加上L2正则化项
            l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            print("l shape: ", l.shape)
            l = l.sum()
            # 清除之前的梯度
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            # 执行反向传播，自动微分
            l.backward()
            # 调用 d2l.sgd 梯度下降，更新权重和偏置
            d2l.sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features, w, b), train_labels).mean().item())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss', range(1, num_epochs + 1), test_ls,
                 ['train', 'test'])
    d2l.plt.show()
    print('L2 form of w:', w.norm().item())

# 过拟合
# fit_and_plot(lambd = 0)
# 权重衰减
# fit_and_plot(lambd = 3)

# 简洁实现
def fit_and_plot_pytorch(wd):
    # 对权重参数衰减
    net = nn.Linear(num_inputs, 1)
    nn.init.normal_(net.weight, mean = 0, std = 1)
    nn.init.normal_(net.bias, mean = 0, std = 0.1)
    optimizer_w = torch.optim.SGD(params = [net.weight], lr = lr, weight_decay = wd)
    optimizer_b = torch.optim.SGD(params = [net.bias], lr = lr)
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y).mean()
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()
            l.backward()
            optimizer_w.step()
            optimizer_b.step()
        train_ls.append(loss(net(train_features), train_labels).mean().item())
        test_ls.append(loss(net(test_features), test_labels).mean().item())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss', range(1, num_epochs + 1), test_ls,
                 ['train', 'test'])
    d2l.plt.show()
    print('L2 form of w:', net.weight.data.norm().item())

# 过拟合
# fit_and_plot_pytorch(wd = 0)
# 权重衰减
# fit_and_plot_pytorch(wd = 3)
