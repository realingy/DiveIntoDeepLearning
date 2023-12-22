import os
import pandas as pd
import torch

# 2.2.1 读取数据集
os.makedirs(os.path.join('.', 'data'), exist_ok=True)
data_file = os.path.join('.', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n') # 列名
    f.write('NA,Pave,127500\n') # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)
print(data)

# 2.2.2 处理缺失值
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
print("inputs: \n", inputs)
print("outputs: \n", outputs)
# 代码有问题，因为Alley列不是数值型
# inputs = inputs.fillna(inputs.mean())
# 改正, 对于数值型列, 使用均值填充缺失值
inputs['NumRooms'] = inputs['NumRooms'].fillna(inputs['NumRooms'].mean())
# 对于分类型列，例如使用一个特定的字符串或者最常见的值填充
# 例如，这里使用 'Unknown' 填充 'Alley' 列的缺失值
# inputs['Alley'] = inputs['Alley'].fillna('Unknown')
print("inputs: \n", inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# 2.2.3 转换为张量格式
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print("X: \n", X)
print("y: \n", y)