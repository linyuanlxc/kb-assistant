---
title: 机器学习4
tags:
- 机器学习
- Python
categories:
- 机器学习
date: 2023-08-20 00:03:01
description: 用pytorch实现线性回归
---

# 用pytorch实现线性回归(linear regression)

使用梯度下降关键是求出损失函数对于权重的导数

## 步骤

1.	准备数据集
2.	设计模型
3.	构建损失函数、优化器
4.	训练周期

## 训练的步骤

1.	求y^（预测结果）
2.	求损失函数
3.	backward
4.	更新

问题同前几篇blog

```python
import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # 构造对象，包含权重与偏置

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()#可调用
#损失值应该是标量
criterion = torch.nn.MSELoss(size_average=False)  # false为不求平均值，即不除以N
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # lr,学习率

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  # 进行更新

print("w = ", model.linear.weight.item())  # weight是矩阵
print("b = ", model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print("y_pred = ", y_test.data)
```

类名必须用大驼峰命名

要把模型定义成一个类，模型类都要继承于torch.nn.Model