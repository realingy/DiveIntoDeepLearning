import torch

# 2.5 自动微分
# 求导是几乎所有深度学习优化算法的关键步骤。虽然求导的计算很简单，只需要一些基本的微积分。但对于复杂的模型，手工进行更新是一件很痛苦的事情（而且经常容易出错）。

# 深度学习框架通过自动计算导数，即自动微分（automatic differentiation）来加快求导。实际中，根据设计
# 好的模型，系统会构建一个计算图（computational graph），来跟踪计算是哪些数据通过哪些操作组合起来产生输出。
# 自动微分使系统能够随后反向传播梯度。这里，反向传播（backpropagate）意味着跟踪整个计算图，填充关于每个参数的偏导数。

# 2.5.1 一个简单的例子
x = torch.arange(4.0)
x

# 在我们计算y关于x的梯度之前，需要一个地方来存储梯度。重要的是，我们不会在每次对一个参数求导时都分配新的内存。因为我们经常会成千上万次地更新相同的参数，每次都分配新的内存可能很快就会将内存耗尽。
# 注意，一个标量函数关于向量x的梯度是向量，并且与x具有相同的形状。

# 优化代码
# 告诉PyTorch我们想要对这个张量进行梯度跟踪, PyTorch将会记录下所有操作这个张量的计算步骤，以便以后计算梯度。这在神经网络训练中非常重要，因为我们需要梯度来更新模型的权重。
x=torch.arange(4.0,requires_grad=True)
# 打印 x.grad 会得到 None，因为此时还没有进行任何操作来计算梯度。x.grad 保存的是 x 的梯度，但是直到我们对 x 执行某些操作并进行反向传播之后，x.grad 才会包含实际的梯度值
print("x.grad: ", x.grad)

y = 2 * torch.dot(x, x)
print("y: ", y)

# x是一个长度为4的向量，计算x和x的点积，得到了我们赋值给y的标量输出。接下来，通过调用反向传播函数
# 来自动计算y关于x每个分量的梯度，并打印这些梯度。

y.backward()
print("x.grad: ", x.grad)

# 函数y = 2x⊤x关于x的梯度应为4x。让我们快速验证这个梯度是否计算正确。
print("x.grad == 4*x: ", x.grad == 4 * x)

x.grad.zero_() # 梯度清零，否则后续迭代的梯度会累积到前面的梯度中
y = x.sum()
y.backward()
print("x.grad of y = x.sum(): ", x.grad)

# 2.5.2 非标量变量的反向传播
# 当y不是标量时，向量y关于向量x的导数的最自然解释是一个矩阵。对于高阶和高维的y和x，求导的结果可以是一个高阶张量。
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
print("2.5.2 x: ", x)
x.grad.zero_()
y = x * x # Hadamard积.逐个元素积
print("2.5.2 y: ", y)
# 等价于y.backward(torch.ones(len(x))),y此时不是一个标量，不能直接反向传播（梯度不是唯一的）,但是可以将y先转成一个标量或者传入一个相同形状的“梯度的种子”
y.sum().backward()
print("2.5.2 x.grad: ", x.grad)

# 2.5.3 分离计算
# 有时，我们希望将某些计算移动到记录的计算图之外。例如，假设y是作为x的函数计算的，而z则是作为y和x的
# 函数计算的。想象一下，我们想计算z关于x的梯度，但由于某种原因，希望将y视为一个常数，并且只考虑
# 到x在y被计算后发挥的作用。
x.grad.zero_()
y = x * x
# 创建了 y 的一个副本，但是 u 不跟踪梯度。.detach() 方法创建了一个新张量，它与原张量共享数据但不参与梯度计算。这意味着对 u 的任何操作都不会在计算图中留下痕迹，因此不会影响梯度的计算
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
# 注意f(a)一定是a的一个线性变换
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
