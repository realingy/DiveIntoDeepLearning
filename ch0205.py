import torch

# 2.5 自动微分
# 求导是几乎所有深度学习优化算法的关键步骤。虽然求导的计算很简单，只需要一些基本的微积分。但对于复杂的模型，手工进行更新是一件很痛苦的事情（而且经常容易出错）。

# 深度学习框架通过自动计算导数，即自动微分（automatic differentiation）来加快求导。实际中，根据设计
# 好的模型，系统会构建一个计算图（computational graph），来跟踪计算是哪些数据通过哪些操作组合起来产生输出。
# 自动微分使系统能够随后反向传播梯度。这里，反向传播（backpropagate）意味着跟踪整个计算图，填充关于每个参数的偏导数。

# 2.5.1 一个简单的例子
x=torch.arange(4.0,requires_grad=True)
x.grad

y = 2 * torch.dot(x, x)
y

# x是一个长度为4的向量，计算x和x的点积，得到了我们赋值给y的标量输出。接下来，通过调用反向传播函数
# 来自动计算y关于x每个分量的梯度，并打印这些梯度。

y.backward()
x.grad

# 函数y = 2x⊤x关于x的梯度应为4x。让我们快速验证这个梯度是否计算正确。
x.grad == 4 * x

x.grad.zero_()
y = x.sum()
y.backward()
x.grad

# 2.5.2 非标量变量的反向传播
# 当y不是标量时，向量y关于向量x的导数的最自然解释是一个矩阵。对于高阶和高维的y和x，求导的结果可以是一个高阶张量。
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
x.grad

# 2.5.3 分离计算
# 有时，我们希望将某些计算移动到记录的计算图之外。例如，假设y是作为x的函数计算的，而z则是作为y和x的
# 函数计算的。想象一下，我们想计算z关于x的梯度，但由于某种原因，希望将y视为一个常数，并且只考虑
# 到x在y被计算后发挥的作用。
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print("x.grad == u: ", x.grad == u)

x.grad.zero_()
y.sum().backward()
print("x.grad == 2 * x: ", x.grad == 2 * x)

# 2.5.4 Python控制流的梯度计算
# 使用自动微分的一个好处是：即使构建函数的计算图需要通过Python控制流（例如，条件、循环或任意函数调用），我们仍然可以计算得到的变量的梯度。
# 在下面的代码中， while循环的迭代次数和if语句的结果都取决于输入a的值
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print("a.grad == d / a: ", a.grad == d / a)
