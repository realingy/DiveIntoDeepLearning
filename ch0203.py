# CH203 线性代数
import torch

# 2.3.1 标量
x = torch.tensor(3.0)
y = torch.tensor(2.0)
print("x: ", x, ", y: ", y)
print("x + y: ", x + y, ", x * y: ", x * y, ", x / y: " , x / y, ", x ** y: " , x**y)

# 2.3.2 向量
x = torch.arange(4)
print("x: ", x)

# 长度、维度和形状
print("len(x): ", len(x))
print("x shape: ", x.shape)

# 2.3.3 矩阵
A = torch.arange(20).reshape(5, 4)
print("A: \n", A)
print("AT: \n", A.T)
# print("A shape: ", A.shape)
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print("B: \n", B)
print("BT: \n", B.T)
# print("B shape: ", B.shape)
print("B == BT: \n", B == B.T)

# 2.3.4 张量
X = torch.arange(24).reshape(2, 3, 4)
print("X: \n", X)

# 2.3.5 张量算法的基本性质
# 任何按元素的一元运算都不会改变其操作数的形状
# 同样，给定具有相同形状的任意两个张量，任何按元素二元运算的结果都将是相同形状的张量
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone() # 通过分配新内存，将A的一个副本分配给B
A, A + B
A * B

# 将张量乘以或加上一个标量不会改变张量的形状，其中张量的每个元素都将与标量相加或相乘。
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape

# 2.3.6 降维
# 可以对任意张量进行的一个有用的操作是计算其元素的和
x = torch.arange(4, dtype=torch.float32)
x, x.sum()
A.shape, A.sum()

# 求和所有行的元素来降维（轴0）
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
# 指定axis=1将通过汇总所有列的元素降维（轴1）
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
# 沿着行和列对矩阵求和，等价于对矩阵的所有元素进行求和
A.sum(axis=[0, 1]) # 结果和A.sum()相同
# 相等
A.mean(), A.sum() / A.numel()
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
A.mean(axis=1), A.sum(axis=1) / A.shape[1]
# 保持轴数不变
sum_A = A.sum(axis=1, keepdims=True)
# 由于sum_A在对每行进行求和后仍保持两个轴，我们可以通过广播将A除以sum_A
A / sum_A
# 沿某个轴计算A元素的累积总和，此函数不会沿任何轴降低输入张量的维度。
A.cumsum(axis=0)

# 2.3.7 点积（Dot Product）
# 定义：两个向量的点积是它们对应元素乘积的总和
# 结果：点积的结果是一个标量
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)
# 等效于点积
torch.sum(x * y)

# 2.3.8 矩阵-向量积
# 定义：两个三维向量的向量积是一个新的向量，其方向垂直于这两个向量构成的平面。
# 结果：向量积的结果是一个向量。
# 公式：如果有两个向量 a = [a1, a2, a3], b = [b1, b2, b3], 它们的向量积是 
# a×b=[a2*b3 − a3*b2, a3*b1 − a1*b3, a1*b2 − a2*b1]
A.shape, x.shape, torch.mv(A, x)

# 2.3.9 矩阵-矩阵乘法
# 定义：矩阵乘法是两个矩阵的行与列的点积组合
# 结果：矩阵乘法的结果是一个新的矩阵
B = torch.ones(4, 3)
torch.mm(A, B)

# 2.3.10 范数
# 线性代数中最有用的一个运算符是范数（norm）。向量的范数是表示一个向量有多大。这里考虑的大小（size）概念不涉及维度，而是分量的大小。
# 在线性代数中，向量范数是将向量映射到标量的函数f
# 给定任意向量x，向量范数要满足一些属性。
# 第一个性质是：如果我们按常数因子α缩放向量的所有元素，其范数也会按相同常数因子的绝对值缩放：
#                               f(αx) = |α|f(x)
# 第二个性质是熟悉的三角不等式:
#                           f(x + y) ≤ f(x) + f(y).
# 第三个性质简单地说范数必须是非负的:
#                                f(x) ≥ 0.

# 1、L1 范数：向量元素绝对值之和
# 2、L2 范数（欧几里得范数）：向量元素平方和的平方根
# 3、向量元素的最大绝对值
u = torch.tensor([3.0, -4.0])
torch.norm(u)
# 深度学习中更经常地使用L2范数的平方，也会经常遇到L1范数
torch.norm(torch.ones((4, 9)))
torch.norm(torch.ones((4, 9)), p=1) # L1范数，默认p=2,即默认求取L2范数
torch.abs(u).sum() # 等效于求L1范数

# 范数和目标
# 在深度学习中，我们经常试图解决优化问题：最大化分配给观测数据的概率; 最小化预测和真实观测之间的距离。
# 用向量表示物品（如单词、产品或新闻文章），以便最小化相似项目之间的距离，最大化不同项目之间的距离。
# 目标，或许是深度学习算法最重要的组成部分（除了数据），通常被表达为范数。

# 标量、向量、矩阵和张量是线性代数中的基本数学对象
# • 向量泛化自标量，矩阵泛化自向量
# • 标量、向量、矩阵和张量分别具有零、一、二和任意数量的轴
# • 一个张量可以通过sum和mean沿指定的轴降低维度
# • 两个矩阵的按元素乘法被称为他们的Hadamard积。它与矩阵乘法不同
# • 在深度学习中，我们经常使用范数，如L1范数、 L2范数和Frobenius范数
# • 我们可以对标量、向量、矩阵和张量执行各种操作