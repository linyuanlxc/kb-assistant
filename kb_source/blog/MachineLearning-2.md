---
title: 机器学习2
date: 2023-08-13 23:32:41
tags: 
- 机器学习
- Python
categories:
- 机器学习
description: 记录有关在pytorch学习中梯度下降算法的知识
---

# 梯度下降算法

---

采用分治的思路，不断划分区块，进行搜索，但是会有可能陷入局部最优解

梯度：目标函数对权重求导。导数为负的方向就是最小指的方向

在单变量的函数中，梯度其实就是函数的微分，代表着函数在某个给定点的切线的斜率
在多变量函数中，梯度是一个向量，向量有方向，梯度的方向就指出了函数在给定点的上升最快的方向

即：`ω = ω + α*θ`。θ为梯度，α为学习率或步长

类似于贪心算法，只看眼前最好的选择，不一定能得到最优结果，但能得到局部最优结果

鞍点：导数为零 

---

## 随机梯度下降算法

梯度下降衍生版本：随机梯度下降（stochastic gradient descent）。随机选择单个样本的损失函数求导。可以避免陷入鞍点。SGD算法是从样本中随机抽出一组，训练后按梯度更新一次，然后再抽取一组，再更新一次，在样本量及其大的情况下，可能不用训练完所有的样本就可以获得一个损失值在可接受范围之内的模型了。

---

针对于MachineLearning-1的问题，梯度下降算法代码如下

```python
import numpy as np
import matplotlib.pyplot as plt

# 数据集
x_data = {1.0, 2.0, 3.0}
y_data = {2.0, 4.0, 6.0}

# 初始权重猜测
w = 1.0


# 定义模型
def forward(x):
    return x * w


# 定义损失函数
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)


def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


print('Predict (before training)', 4, forward(4))
for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val
    print('Epoch:', epoch, 'w=', w, 'loss=', cost_val)
print('Predict (after training)', 4, forward(4))
```

随机梯度下降代码

```python
import numpy as np
import matplotlib.pyplot as plt

# 数据集
x_data = {1.0, 2.0, 3.0}
y_data = {2.0, 4.0, 6.0}

# 初始权重猜测
w = 1.0


# 定义模型
def forward(x):
    return x * w


# 定义损失函数
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


def gradient(xs, ys):
    return 2 * x * (x * w - y)


print('Predict (before training)', 4, forward(4))
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x_data, y_data)
        w -= 0.01 * grad
        print('\tgrad:', x, y, grad)
    print('progress:', epoch, 'w=', w, 'loss=', loss(x, y))
print('Predict (after training)', 4, forward(4))
```

梯度下降算法可以并行化，随机梯度下降不行

mini-batch：小批量随机梯度下降。如果数据样本的数量很大，那么一轮的迭代会很耗时间，所以可以把这些数据样本划分为若干份，这些子集就叫做mini—batch

有关随机梯度下降文章
> https://blog.csdn.net/qq_58146842/article/details/121280968?ops_request_misc=&request_id=&biz_id=102&utm_term=%E9%9A%8F%E6%9C%BA%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D%E7%AE%97%E6%B3%95%E5%8E%9F%E7%90%86&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-121280968.nonecase&spm=1018.2226.3001.4187