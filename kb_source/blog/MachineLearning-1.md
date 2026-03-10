---
title: 机器学习1
date: 2023-08-12 21:50:40
tags: 
- 机器学习
- Python
categories:
- 机器学习
description: 记录有关在pytorch学习中线性模型的知识和一些基本概念
---

# 一些基本概念

## 张量

pytorch中的基本数据结构，可以理解为多维数组。

```python
import torch

a = torch.ones(3)  # 创建一个大小为3的一维张量，用1.0填充
print(type(a))
a[2] = 55
print(a)
```

```python
a = torch.tensor([[2, 3], [3, 4]])  # 创建一个2维张量
print(type(a))
print(a)
a[0, 0] = 55
print(a)
print(a.shape)  #查看每个维度上张量的大小
```

可以通过从0开始的索引来访问张量中的每一个元素，也可以修改值。

切片的方法也适用于张量。

```python
print(a[1:])    #第一行之后的所有行，所有列
print(a[1:,:])  #第一行之后的所有行，所有列
print(a[1:,0])  #第一行之后的所有行，第一列
print(a[None])  #增加大小为一的维度，类似于unsqueeze()方法
```

**张量的大小、偏移量、步长**。张量的大小是一个元组，表示张量在每个维度上有多少个元素；偏移量是指存储区中某个元素相对于张量中第一个元素的索引；步长是指存储区中为了获得下一个元素需要跳过的元素数量，它是一个元组，指示当索引在每个维度增加1时在储存区中要跳过的元素数量。

```python
import torch

a = torch.tensor(
    [[[3, 2, 1], [1, 6, 7], [2, 6, 8]], [[13, 32, 11], [41, 46, 7], [52, 65, 84]]]
)
print(a.size())
point = a[1, 2, 2]
print(point)
print(point.storage_offset())  # 该元素相对于第一个元素的偏移
print(a.stride())  # 步长
print(a.dim())  #查看张量的维度
```

需要注意的是，通过索引的方式去更改子张量，会影响原始张量，这是因为子张量与原始张量索引了相同的存储区。可以使用clone()方法复制新的张量。

```python
import torch

a = torch.tensor(
    [[[3, 2, 1], [1, 6, 7], [2, 6, 8]], [[13, 32, 11], [41, 46, 7], [52, 65, 84]]]
)
print(a.size())
point = a[1, 2]
print(point)
point[1] = 111
print(a)
point1 = point.clone()  # 复制新的张量
print(point1)
point1[1] = 222
print(point1)
print(a)
```

此外，张量还有其他操作。如：转置t()方法、连续contiguous()方法。张量还具有设备属性，可以指定在CPU或者GPU上创建张量。可以将张量保存(save()方法)，也可以加载本地张量(load()方法)。可以通过`h5py`库，将张量转化成NumPy数组，这样可以适用于不同的库。

# 线性模型

## 机器学习(machine learning)几个步骤 
1. **准备数据集(dataset)**
   
   将我们的数据转变成张量(tensor)，一般使用torch.utils.data包下的Dataset类中的API接口。还有加载数据集，一般是使用DataLoader类。有时还需要对数据集进行预处理，如下。
    
    ```python
    form torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ]
    )
    ```

    这个预处理函数的意思是：将输入图像缩放到256*256，围绕中心将图像裁剪为224*224个像素，将其转换成张量，对其RGB分量进行归一化处理，使其具有定义的均值和标准差。

2. **选择模型(model)。例如pytorch提供的cv相关的模型**
   
   ```python
   from torchvision import models
   print(dir(models))
   ```
    通过上述代码可以查看cv相关的模型。

3. **训练(training)**
4. **应用(inferring)**

数据集分为两部分：训练集、测试集。

训练集又可以细分为训练集、开发集

损失函数是针对一个样本的,平均平方误差(Mean Square Error,mse)是针对于整个训练集。

训练神经网络本质上就是使用几个或者一些参数将一个模型变换为更加复杂的模型

损失函数的选择很重要，因为它是一种对训练样本中要修正的错误进行优先处理的方法，可以强调或者忽略某些误差

优化器：torch.optiom中提供，用于更新


## 过拟合

用训练集去训练模型，在尽可能的使损失最小后，将模型在测试集验证时发现，模型产生的损失比预期的要高得多，即过拟合

<div align=center>
<img src="MachineLearning-1-3.jpg" height = '360'>
引自《Deep Learning with PyTorch》
</div>

解决过拟合的方法
1. 在损失函数中添加惩罚项，以降低模型的成本，使其表现得更加平稳、变换更缓慢
2. 在输入样本中添加噪声，人为地在训练数据样本之间创建新的数据点，并使模型也拟合这些点
3. ···

那么现在可以将训练神经网络（选择合适参数）的过程分为两步：增大参数直到拟合；缩小参数以避免出现过拟合

## 损失函数

## 练习

### Question

Suppose that students would get y points in final exam, if they spent x hours in study

| x   | y   |
| --- | --- |
| 1   | 2   |
| 2   | 4   |
| 3   | 6   |
| 4   | ?   |

### Answer

```python

import numpy as np
import matplotlib.pyplot as plt
	
# 数据集
x_data = {1.0, 2.0, 3.0}
y_data = {2.0, 4.0, 6.0}

# 定义模型
def forward(x):
    return x * w

# 定义损失函数
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

#权重及其对应损失值
w_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):
    print('w=',w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', l_sum/3)
    w_list.append(w)
    mse_list.append(l_sum/3)

plt.plot(w_list, mse_list)
plt.ylabel('loss')
plt.xlabel('w')
plt.show()
```

<div align=center>
<img src="MachineLearning-1-1.png" height = '360' title="y=x*w" alt="y=x*w">
y=x*w
</div>

---

### Question
Suppose that students would get y points in final exam, if they spent x hours in study.Try to use the model y=x*w+b, and draw the cost graph.

| x   | y   |
| --- | --- |
| 1   | 2   |
| 2   | 4   |
| 3   | 6   |
| 4   | ?   |

### Answer

```python
import numpy as np
import matplotlib.pyplot as plt

# 数据集
x_data = {1.0, 2.0, 3.0}
y_data = {2.0, 4.0, 6.0}


# 定义模型
def forward(x):
    return x * w + b


# 定义损失函数
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# 权重及其对应损失值
w_list = []
b_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):
    for b in np.arange(-2.0, 2.0, 0.1):
        print('w=',  w, 'b=', b)
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
            print('\t', x_val, y_val, y_pred_val, loss_val)
        print('MSE=', l_sum/3)
        w_list.append(w)
        b_list.append(b)
        mse_list.append(l_sum/3)

print(mse_list)
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  # 创建三维坐标系
ax3d.plot_trisurf(w_list, b_list, mse_list)
plt.show()
```

<div align=center>
<img src="MachineLearning-1-2.png" height = '360' title="y=x*w+b" alt="y=x*w+b">
y=x*w+b
</div>

---

matplotlib中的函数还不怎么会用，后面抽个时间看一看

> https://blog.csdn.net/hustlei/article/details/122408179

这个博客里面思维导图可以看一看捏
