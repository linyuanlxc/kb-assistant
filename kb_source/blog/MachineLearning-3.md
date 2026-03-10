---
title: 机器学习3
date: 2023-08-17 21:43:42
tags: 
- 机器学习
- Python
categories:
- 机器学习
description: 记录有关反向传播算法的知识
---

# 反向传播（Back Propagation）

权重太多，求解析式十分复杂，因此在面对复杂网络时，尝试把网络看作是一个图，根据链式法则(chain rule)，求其解析式，即反向传播算法。

Back Propagation链式求导过程
1. Create Computational Graph
2. Local Gradient
3. Given gradient from successive node
4. Use chain rule to compute the gradient(Backward) 

Tensor 张量，pytorch里的重要组成，可以是标量，也可以是向量，矩阵。它包含数据（data），导数（grad），即权重与损失函数对权重的导数。

正向的目标是求出本次的损失，反向的目标则是求出导数。蓝色为正向，红色为反向

<div align=center>
<img src="MachineLearning-3-1.png" height = '360'>
</div>

```python
import torch

# 数据集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 创建一个Tensor变量，权重
w = torch.Tensor([1.0])
# 需要计算梯度
w.requires_grad = True


# 定义模型，此时w是Tensor，*要计算Tensor与Tensor之间的乘法，所以有一个对x的自动类型强转
def forward(x):
    return x * w


# 定义损失函数，构建计算图
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


print("predict (before training)", 4, forward(4).item())
# w.grad也是个梯度，w.grad.item才是标量
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)  # l是个张量，前馈过程，计算loss
        l.backward()#计算梯度
        print("\tgrad:", x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data#梯度用于更新权重
        w.grad.data.zero_()  # 梯度清零

    print("process:", epoch, l.item())

print("predict(after training)", 4, forward(4).item())
```
需要注意的是每一次通过.backward()计算的梯度会累积，因此在更新后，需要通过.grad.data.zero()将梯度置零

权重是指w，梯度是指loss对w的导数。要先通过前向过程求出loss，求出loss后通过反向传播求出loss对w的导数,再根据随机梯度下降算法或者其他算法对权重w进行更新。