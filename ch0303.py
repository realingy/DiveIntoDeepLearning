import numpy as np
import torch
from torch.utils import data

# 3.3 线性回归的简洁实现

# 3.3.1 生成数据集
def synthetic_data(w, b, num_examples): #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# 3.3.2 读取数据集
def load_array(data_arrays, batch_size, is_train=True): #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 3.3.3 定义模型
# nn是神经网络的缩写
from torch import nn
# 我们只需关注使用哪些层来构造模型，而不必关注层的实现细节。我们首先定义一个模型变量net，它是一个Sequential类的实例。
# Sequential类将多个层串联在一起。当给定输入数据时， Sequential实例将数据传入到第一层，然后将第一层的输出作为第二层的输入，以此类推
net = nn.Sequential(nn.Linear(2, 1))

# 3.3.4 初始化模型参数
# 指定每个权重参数应该从均值为0、标准差为0.01的正态分布中随机采样，偏置参数将初始化为零
# 正如我们在构造nn.Linear时指定输入和输出尺寸一样，现在我们能直接访问参数以设定它们的初始值。
# 我们通过net[0]选择网络中的第一个图层，然后使用weight.data和bias.data方法访问参数。我们还可以使用替换方法normal_和fill_来重写参数值。
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 3.3.5 定义损失函数
# 计算均方误差使用的是MSELoss类，也称为平方L2范数。默认情况下，它返回所有样本损失的平均值
# 回归任务常用的损失函数有均方误差（MSE），分类任务常用的有交叉熵损失(Cross-Entropy)
loss = nn.MSELoss()

# 3.3.6 定义优化算法
# 小批量随机梯度下降算法是一种优化神经网络的标准工具， PyTorch在optim模块中实现了该算法的许多变种。
# 当我们实例化一个SGD实例时，我们要指定优化的参数（可通过net.parameters()从我们的模型中获得）以及优化算法所需的超参数字典。
# 小批量随机梯度下降只需要设置lr值，这里设置为0.03。
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 3.3.7 训练
# 通过深度学习框架的高级API来实现我们的模型只需要相对较少的代码
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差: ', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差: ', true_b - b)

# 如何将小批量的总损失表达为每个批量里面损失的平均值？
# 在小批量训练中，总损失通常是批量中所有样本损失的总和。要得到平均损失，我们可以将总损失除以批量中的样本数。