# 2.4 微积分
# 在深度学习中，我们“训练”模型，不断更新它们，使它们在看到越来越多的数据时变得越来越好。通常情
# 况下，变得更好意味着最小化一个损失函数（loss function），即一个衡量“模型有多糟糕”这个问题的分数。
# 最终，我们真正关心的是生成一个模型，它能够在从未见过的数据上表现良好。但“训练”模型只能将模型
# 与我们实际能看到的数据相拟合。因此，我们可以将拟合模型的任务分解为两个关键问题：
#       优化（optimization）：用模型拟合观测数据的过程；
#       泛化（generalization）：数学原理和实践者的智慧，指导我们生成出有效性超出用于训练的数据集本身的模型。

# 2.4.1 导数和微分
# 我们首先讨论导数的计算，这是几乎所有深度学习优化算法的关键步骤。在深度学习中，我们通常选择对于模型参数可微的损失函数。
# 简而言之，对于每个参数，如果我们把这个参数增加或减少一个无穷小的量，可以知道损失会以多快的速度增加或减少

import numpy as np
import torch

def f(x):
    return 3 * x ** 2 - 4 * x

def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

h = 0.1
for i in range(10):
    print(f'h={h:.10f}, numerical limit={numerical_lim(f, 1, h):.10f}')
    h *= 0.1

# 2.4.2 偏导数
# 为了计算 ∂y/∂xi ，我们可以简单地将x1, . . . , xi−1, xi+1, . . . , xn看作常数，并计算y关于xi的导数

# 2.4.3 梯度
# 我们可以连结一个多元函数对其所有变量的偏导数，以得到该函数的梯度（gradient）向量
# 输入是一个n维向量x, 输出是一个标量, 函数f(x)相对于x的梯度是一个包含n个偏导数的向量
# 梯度对于设计深度学习中的优化算法有很大用处

# 2.4.4 链式法则
# 上面方法可能很难找到梯度。这是因为在深度学习中，多元函数通常是复合（composite）的，所以难以应用上述任何规则来微分这些函数。幸运的是，链式法则可以被用来微分复合函数。
