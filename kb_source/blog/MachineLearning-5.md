---
title: 机器学习5
tags:
- 机器学习
- Python
categories:
- 机器学习
mathjax: true
date: 2023-08-21 00:36:56
description: 逻辑斯蒂回归
---

# 逻辑斯蒂回归（logistic regression)

<div align=center>
<img src="MachineLearning-5-2.png" height = '360'>
</div>

名字虽然是回归，但实际上是分类

回归问题的预测结果范围是实数，而分类问题的预测结果为0~1，因此需要映射

可以使用Sigmoid函数(s型函数)，将线性模型的结果压缩，比如：

$$
f(x)=\frac{1}{1+{\rm e}^{-x}}
$$

那么预测结果(0~1)就为

$$
\hat{y} = f(w·x+b)=\frac{1}{1+{\rm e}^{-(w·x+b)}}
$$

<div align=center>
<img src="MachineLearning-5-1.png" height = '360'>
</div>

<!--
## 逻辑斯蒂回归模型

<div align=center>
<img src="MachineLearning-5-3.png" height = '360'>
</div>

最后的判别结果是通过比较P(Y=1|x)和P(Y=0|x)的大小来确定的，若Y=1大，那么Y=1，反之Y=0

<p /hidden>
## 损失函数
<div align=center>
<img src="MachineLearning-5-5.png" height = '360'>
</div>

<div align=center>
<img src="MachineLearning-5-4.png" height = '360'>
</div>

### 对数似然函数

似然函数可以理解成通过已知的结果去倒推出最大概率得到该结果的参数，即“模型已定，参数未知”。就比如已知一些数据，我们对这个模型使用正态分布，那么我们就可以求出与已知的数据对应最好的那条拟合曲线，求出均值μ，标准差σ。

对于逻辑斯蒂回归,近似看成二项分布

$$
p(Y=Y_i|x_i,w) = 
\begin{cases}
[π(x_i)]^{Y_i}		& Y_i=1 \\
[1-π(x_i)]^{1-Y_i}]	& Y_i=0	\\
\end{cases}
$$

则有
$$
p(Y=Y_i|x_i,w) = π(x_i)^{Y_i}·[1-π(x_i)]^{1-Y_i}]
$$

当有m个样本时
$$
P = \prod_{i=1}^m [π(x_i)^{Y_i}·[1-π(x_i)]^{1-Y_i}]
$$

对其取对数

$$
\begin{aligned}
L(w) &= \log_{e}{\prod_{i=1}^m [π(x_i)^{Y_i}·[1-π(x_i)]^{1-Y_i}]} \\
&= \sum_{i=1}^m [ {Y_i} log_{e}π(x_i) + (1-Y_i)log_{e}[1-π(x_i)] ] 
\end{aligned}
$$
-->

## 损失函数

<div align=center>
<img src="MachineLearning-5-6.png" height = '360'>
</div>

损失函数不再使用MSE，采用下面的式子

- 当预测结果 $\hat{y}=1$，实际结果为$1$时，$loss=0$
- 当预测结果 $\hat{y}=0$，实际结果为$1$时，$loss=+\infty$
- 当预测结果 $\hat{y}=1$，实际结果为$0$时，$loss=-\infty$
- 当预测结果 $\hat{y}=0$，实际结果为$0$时，$loss=0$

## 代码
```python
import torch.nn.functional as F
import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])


class LogisticRegreesionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegreesionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # 构造对象，包含权重与偏置

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegreesionModel()

criterion = torch.nn.BCELoss(size_average=False)  # false为不求平均值，即不除以N
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 优化器 lr,学习率

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  # 进行更新

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print("y_pred = ", y_test.data)
```

如果用matplotlib绘图，也可以看见输出结果$\hat{y}$为S型曲线
