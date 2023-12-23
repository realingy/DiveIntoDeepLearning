import math
import time
import numpy as np
import torch

class Timer: #@save
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()
    
    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

# 3.1 线性回归

# 回归（regression）是能为一个或多个自变量与因变量之间关系建模的一类方法。在自然科学和社会科学领域，回归经常用来表示输入和输出之间的关系

# 解析解（analytical solution）
# 线性回归的解可以用一个公式简单地表达出来，这类解叫作解析解（analytical solution）,但是绝大多数的非线性回归问题都不存在解析解，解析解可以进行很好的数学分
# 析，但解析解对问题的限制很严格，导致它无法广泛应用在深度学习里

# 随机梯度下降（gradient descent）
# 即使在我们无法得到解析解的情况下，我们仍然可以有效地训练模型，这种方法几乎可以优化所有深度学习模型，它通过不断地在损失函数递减的方向上更新参数来降低误差。
# （1）初始化模型参数的值，如随机初始化；
# （2）从数据集中随机抽取小批量样本且在负梯度的方向上更新参数，并不断迭代这一步骤。

# 3.1.2 矢量化加速
# 在训练我们的模型时，我们经常希望能够同时处理整个小批量的样本。为了实现这一点，需要我们对计算进行矢量化，从而利用线性代数库，而不是在Python中编写开销高昂的for循环。
def interval():
    n = 10000
    a = torch.ones([n])
    b = torch.ones([n])

    c = torch.zeros(n)

    # 1. 遍历
    timer = Timer()
    for i in range(n):
        c[i] = a[i] + b[i]
    print("interval1: ", f'{timer.stop():.5f} sec')

    # 2. 重载的+函数
    timer.start()
    d = a + b
    print("interval12: ", f'{timer.stop():.5f} sec')

    # 结果很明显，第二种方法比第一种方法快得多。矢量化代码通常会带来数量级的加速。另外，我们将更多的数学运算放到库中，而无须自己编写那么多的计算，从而减少了出错的可能性。

# interval()

# 3.1.3 正态分布与平方损失
# 接下来，我们通过对噪声分布的假设来解读平方损失目标函数。
# 正态分布和线性回归之间的关系很密切。正态分布（normal distribution），也称为高斯分布（Gaussian distribution），最早由德国数学家高斯（Gauss）应用于天文学研究。简单的说，若随机变量x具有均值µ和方
# 差σ2（标准差σ），其正态分布概率密度函数
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)

# 均方误差损失函数（简称均方损失）可以用于线性回归的一个原因是：我们假设了观测中包含噪声，其中噪声服从正态分布。噪声正态分布如下式:
#                                                     y = w⊤x + b + ϵ,
# 可以通过给定的x观测到特定y的似然(likelihood)，然后根据最大似然估计法，参数w和b的最优值是使整个数据集的似然最大的值

# 3.1.4 从线性回归到深度网络
# 管神经网络涵盖了更多更为丰富的模型，我们依然可以用描述神经网络的方式来描述线性模型，从而把线性模型看作一个神经网络。首先，我们用“层”符号来重写这个模型。
# 对于线性回归，每个输入都与每个输出（在本例中只有一个输出）相连，我们将这种变换称为全连接层（fully‐connected layer）