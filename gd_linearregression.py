import numpy as np

class GDLinearRegression:
    
    # def __init__(self, n_iter=200, eta=1e-3, tol=None):
    def __init__(self, n_iter, eta, tol):
        self.n_iter = n_iter #训练迭代次数
        self.eta = eta #学习率
        self.tol = tol #误差变化阈值
        self.w = None  #初始化模型参数w

    def _loss(self, y, y_pred):
        '''计算损失'''
        return np.sum((y_pred - y) ** 2) / y.size
    
    def _gradient(self, x, y, y_pred):
        '''计算梯度'''
        return np.matmul(y_pred - y, x) / y.size
    
    def _gradient_descent(self, w, x, y):
        '''梯度下降算法'''

        #指定tol：早期停止法
        if self.tol is not None:
            loss_old = np.inf

            #至多迭代n_iter次，更新w
            for step_i in range(self.n_iter):
                y_pred = self._predict(x, w)
                loss = self._loss(y, y_pred)
                print('%4i loss: %s' % (step_i, loss))

                #早期停止法
                if self.tol is not None:
                    if loss_old  - loss < self.tol:
                        break
                    loss_old = loss

                #计算梯度
                grad = self._gradient(x, y, y_pred)
                w -= self.eta * grad

    def _preprocess_data_x(self, x):
        '''数据预处理'''

        m, n = x.shape
        x_ = np.empty((m, n + 1))
        x_[:, 0] = 1
        x_[:, 1:] = x

        return x_
    
    def train(self, x_train, y_train):

        x_train = self._preprocess_data_x(x_train)
        _, n = x_train.shape
        self.w = np.random.random(n) * 0.05

        self._gradient_descent(self.w, x_train, y_train)

    def _predict(self, x, w):
        return np.matmul(x, w)
        #预测内部接口，实现函数h(x)
    
    def predict(self, x):
        x = self._preprocess_data_x(x)
        return self._predict(x, self.w)                                                                                                                                                                                                                                                                                                                                                                                                                                                  