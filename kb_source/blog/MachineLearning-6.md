---
title: 机器学习6
tags:
  - 机器学习
  - Python
categories:
  - 机器学习
mathjax: true
description: 处理多维输入
date: 2023-08-29 20:07:10
---


# 多维输入

对于标量或者一维的回归任务，其回归模型可以表示为$\hat{y}=\sigma(x^{(i)}·w+b)$。其中$\sigma$是激活函数，类似于sigmoid函数，i为第i个样本

但对于多维的数据，例如有这么一个样本

| x1  | x2  | x3  | x4  | y   |
| --- | --- | --- | --- | --- |
| 10  | 11  | 21  | 31  | 234 |
| 10  | 11  | 21  | 31  | 234 |
| 10  | 11  | 21  | 31  | 234 |

对于这么一组数据来说，x1、x2、x3、x4被称为**特征(feature)**。那么回归模型的输入就是一组向量:

$$
\left[
\begin{matrix}
    x_1 & x_2 &x_3 & x_4
\end{matrix}
\right]
$$

回归模型变为$\hat{y}=\sigma(\sum_{n=1}^{4} x{_n^{(i)}}·w_n+b)=\sigma(z^{(i)})$，$x{_n^{(i)}}$表示第i个样本的$x_n$。其中的w·x虽然表示的是标量相乘，但是实际的含义是矩阵相乘，即

$$
\left[
\begin{matrix}
    x_1 & x_2 & x_3 & x_4
\end{matrix}
\right]
\left[
\begin{matrix}
    w_1\\
    w_2\\
    w_3\\
    w_4\\
\end{matrix}
\right]
$$

计算结果依旧是标量

# 对于mini-batch(N samples)

回归模型依旧是$\hat{y}=\sigma(\sum_{n=1}^{N} x{_n^{(i)}}·w_n+b)=\sigma(z^{(i)})$

现在有N个样本

$$
\left[
\begin{matrix}
    {\hat{y}}^{(1)} \\
    \vdots \\
    {\hat{y}}^{(N)} \\
\end{matrix}
\right] =
\left[
\begin{matrix}
    \sigma(z^{(1)}) \\
    \vdots \\
    \sigma(z^{(N)}) \\
\end{matrix}
\right] = 
\sigma
\left[
\begin{matrix}
    z^{(1)} \\
    \vdots \\
    z^{(N)} \\
\end{matrix}
\right] 
$$

$$
z^{(N)}=\left[
\begin{matrix}
    x^{(N)}_1 & \cdots & \cdots & x^{(N)}_4
\end{matrix}
\right] 
\left[
\begin{matrix}
    w_1 \\
    \vdots \\
    \vdots \\
    w_4
\end{matrix}
\right]+b 
$$

计算结果为标量，转置不影响,其中4是feature数量

那么我们就可以得到

$$
\left[
\begin{matrix}
    z^{(1)} \\
    \vdots \\
    z^{(N)} \\
\end{matrix}
\right]=
\left[
\begin{matrix}
    x^{(1)}_1 & \cdots & x^{(1)}_4\\
    \vdots    & \ddots & \vdots   \\
    x^{(N)}_1 & \cdots & x^{(N)}_4
\end{matrix}
\right]
\left[
\begin{matrix}
    w_1 \\
    \vdots \\
    \vdots \\
    w_4
\end{matrix}
\right]+
\left[
\begin{matrix}
    b \\
    \vdots \\
    b\\
\end{matrix}
\right]
$$

设样本数量为N，feature的数量为n，则

| x   | w   | b   |
| --- | --- | --- |
| N×n | n×1 | N×1 |

转成向量化的计算后，可利用并行化计算，提高计算速率。

在改代码时，只需将`self.linear = torch.nn.Linear(1, 1)`改成`self.linear = torch.nn.Linear(n, 1)`

如果是多层网络，那么只需将上一层的输出，作为下一层的输入。

<div align=center>
<img src="MachineLearning-6-1.png" height = '360'>
</div>

# 实际问题

<div align=center>
<img src="MachineLearning-6-2.png" height = '360'>
</div>

X1~X8为糖尿病病人的一系列指标，Y表示一年后病情是否加重。

```python
import numpy as np
import torch


xy = np.loadtxt(
    "D:\BaiduNetdiskDownload\PyTorch深度学习实践\diabetes.csv.gz",
    delimiter=",",
    dtype=np.float32,
)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)  # 8维到6维
        self.linear2 = torch.nn.Linear(6, 4)  # 6维到4维
        self.linear3 = torch.nn.Linear(4, 1)  # 4维到1维
        self.activate = torch.nn.Sigmoid()
        # self.activate=torch.nn.ReLU()#采用不同的激活函数

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        return x


model = Model()

criterion = torch.nn.BCELoss(size_average=True)  # false为不求平均值，即不除以N
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # lr,学习率

for epoch in range(1000):
    # 前馈
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    # 反馈
    optimizer.zero_grad()
    loss.backward()
    # 更新
    optimizer.step()  # 进行更新


x_test = torch.Tensor([[-0.29, 0.49, 0.18, -0.29, 0.00, 0.00, -0.53, -0.03]])

y_test = model(x_test)
print("y_pred = ", y_test.data)
```