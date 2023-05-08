import numpy as np

class OLSLinearRegression:

    def _ols(self, x, y):
        '''最小二乘法估算w'''
        # tmp = np.linalg.inv(np.matmul(x.T, x))
        # tmp = np.matmul(tmp, x.T)
        return np.linalg.inv(x.T @ x) @ x.T @ y
    
    def _preprocess_data_X(self, x):
        '''数据预处理'''

        m, n = x.shape
        x_ = np.empty((m, n + 1))
        x_[:, 0] = 1
        x_[:, 1:] = x

        return x_
    
    def train(self, x_train, y_train):
        '''训练模型'''

        x_train = self._preprocess_data_X(x_train)

        self.w = self._ols(x_train, y_train)

    def predict(self, x):
        '''预测'''

        x = self._preprocess_data_X(x)
        return np.matmul(x, self.w)