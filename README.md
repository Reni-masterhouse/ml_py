# ml_py
machine learning in python

第一章：线性回归

我们假设模型形如 h(x)=w^T * x + b，为了方便使用矩阵计算，我们使用 x=(1, x1, x2, ……, xn)^T 和 w=(b, w1, w2, ……， wn)^T 以方便函数运算。

1.1 最小二乘法

使用均方误差（MSE）作为损失函数，对J(w)求梯度为0可得 w=(x^Tx)^(-1)x^Ty。
np.linalg.inv()：求np逆矩阵
np.matmul：矩阵乘法

1.2 梯度下降

每步都沿梯度反方向前进一定学习率大小的步长，参数更新公式为 w:=w-eta*(1/m)*x^T(xw-y)。

1.3 项目 - 预测红酒口感
调用sklearn.model_selection中的train_test_split(x, y, test_size=0.3)切分数据集为训练和测试集。

MSE: mean_squared_error
MAE: mean_absolute_error

第二章 Logistic回归与Softmax回归
