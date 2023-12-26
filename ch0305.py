import torch
import torchvision
from torch.utils import data
from torchvision import transforms

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="./data_mnist", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="./data_mnist", train=False, transform=trans, download=True)

print("len train: ", len(mnist_train), ", len test: ", len(mnist_test))
print("mnist[0][0] shape: ", mnist_train[0][0].shape)

# Fashion‐MNIST是一个服装分类数据集，由10个类别的图像组成。我们将在后续章节中使用此数据集来评估各种分类算法
# Fashion‐MNIST中包含的10个类别，分别为t‐shirt（T恤）、 trouser（裤子）、 pullover（套衫）、 dress（连衣裙）、 coat（外套）、 sandal（凉鞋）、 shirt（衬衫）、 sneaker（运动鞋）、 bag（包）和ankle boot（短靴）。
def get_fashion_mnist_labels(labels): #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

# 3.5.2 读取小批量
# 为了使我们在读取训练集和测试集时更容易，我们使用内置的数据迭代器，而不是从零开始创建。回顾一下，在每次迭代中，数据加载器每次都会读取一小批量数据，大小为batch_size。通过内置数据迭代器，我们可以随机打乱所有样本，从而无偏见地读取小批量。
batch_size = 256
def get_dataloader_workers(): #@save
    """使用4个进程来读取数据"""
    return 4

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())

# 3.5.3 整合所有组件
# 现在我们定义load_data_fashion_mnist函数，用于获取和读取Fashion‐MNIST数据集。这个函数返回训练集和验证集的数据迭代器。此外，这个函数还接受一个可选参数resize，用来将图像大小调整为另一种形状。
def load_data_fashion_mnist(batch_size, resize=None): #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="./data_mnist", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data_mnist", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()), data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))

train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break

# if __name__ == '__main__':
    # freeze_support()