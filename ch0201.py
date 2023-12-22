import torch

# 2.1 
# 2.1.1 基础
x = torch.arange(12)
print("x: \n", x)
print("x.shape: ", x.shape)
print("x.size: ", x.numel())

# X = x.reshape(3, 4)
# X = x.reshape(3, -1)
X = x.reshape(-1, 4)
print("X: \n", X)

print("torch zeros: \n", torch.zeros((2, 3, 4)))
print("torch ones: \n", torch.ones((2, 3, 4)))
print("torch randn: \n", torch.randn((2, 3, 4)))
print("torch xx: \n", torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))

# 2.1.2 运算符
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print("x + y: \n", x + y)
print("x - y: \n", x - y)
print("x * y: \n", x * y)
print("x / y: \n", x / y)
print("x ** y: \n", x ** y)

print("exp(x): \n", torch.exp(x))

X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print("cat1: \n", torch.cat((X, Y), dim=0))
print("cat2: \n", torch.cat((X, Y), dim=1))
print("X == Y: \n", X == Y)
print("X > Y: \n", X > Y)
print("X < Y: \n", X < Y)
print("X.sum(): \n", X.sum())

# 2.1.3 广播机制
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print("a: \n", a)
print("b: \n", b)
print("a + b: \n", a + b)

# 2.1.4 索引和切片
print("X[-1]: \n", X[-1])
print("X[1:3]: \n", X[1:3])
print("X: \n", X)

X[1, 2] = 9
print("X: \n", X)

X[0:2, :] = 12
print("X: \n", X)

# 2.1.5 节省内存
before = id(Y)
print("before: ", before)
Y = Y + X # 会重新分配内存
print("id(Y) == before: ", id(Y) == before)

Z = torch.zeros_like(Y)
print("id(Z): ", id(Z))
print("Z: \n", Z)
Z[:] = X + Y # 不会重新分配内存
Z = X + Y # 会重新分配内存
print("id(Z): ", id(Z))
print("Z: \n", Z)

before = id(X)
X += Y # 不会重新分配内存
# X[:] = X + Y # 不会重新分配内存
# X = X + Y # 会重新分配内存
print("id(X) == before: ", id(X) == before)

# 2.1.6 对象转换
A = X.numpy()
B = torch.tensor(A)
print("type(A): ", type(A), ", type(B): ", type(B))

# 形状大小为1的张量转python标量
a = torch.tensor([3.5])
print("a: ", a, ", a.item(): ", a.item(), ", float(a): ", float(a), ", int(a): ", int(a))