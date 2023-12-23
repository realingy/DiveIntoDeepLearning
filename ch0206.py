import torch
from torch.distributions import multinomial

# 2.6 概率
# 简单地说，机器学习就是做出预测

# 2.6.1 基本概率论
# 假设我们掷骰子，想知道看到1的几率有多大，而不是看到另一个数字。如果骰子是公平的，那么所有六个结果{1, . . . , 6}都有相同的可能发生，因此我们可以说1发生的概率为1/6。

fair_probs = torch.ones([6]) / 6
count = multinomial.Multinomial(1000, fair_probs).sample()
print("count/1000: ", count/1000)

# 进行500组实验，每组抽取10个样本
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
print("cum_counts: ", estimates)

# 概率论公理
# 在处理骰子掷出时，我们将集合S = {1, 2, 3, 4, 5, 6} 称为样本空间（sample space）或结果空间（outcome space），其中每个元素都是结果（outcome）。
# 事件（event）是一组给定样本空间的随机结果。例如，“看到5”（{5}）和“看到奇数”（{1, 3, 5}）都是掷出骰子的有效事件。注意，如果一个随机实验的结果在A中，则
# 事件A已经发生。也就是说，如果投掷出3点，因为3 ∈ {1, 3, 5}，我们可以说，“看到奇数”的事件发生了。
# 概率（probability）可以被认为是将集合映射到真实值的函数。在给定的样本空间S中，事件A的概率，表示为P (A)，满足以下属性：
#   • 对于任意事件A，其概率从不会是负数，即P (A) ≥ 0；
#   • 整个样本空间的概率为1，即P(S) = 1；
#   • 对于互斥（mutually exclusive）事件的任意一个可数序列中任意一个事件发生的概率等于它们各自发生的概率之和

# 随机变量
# 在我们掷骰子的随机实验中，我们引入了随机变量（random variable）的概念。随机变量几乎可以是任何数量，并且它可以在随机实验的一组可能性中取一个值。

