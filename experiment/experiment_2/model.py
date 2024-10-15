import numpy as np
class BayesianClassifier:
    def __init__(self):
        self.w = None
        self.b = None


    def fit(self,X_train,y_train):
        """
        训练模型
        :param X_train: 训练数据
        :param y_train: 训练标签
        """
        class_0 = X_train[y_train == 0]
        class_1 = X_train[y_train == 1]
        mean_0 = np.mean(class_0, axis=0)
        mean_1 = np.mean(class_1, axis=0)
        cov_0 = np.cov(class_0.T)
        cov_1 = np.cov(class_1.T)
        cov = (cov_0 * len(class_0) + cov_1 * len(class_1)) / (len(class_0) + len(class_1))

        self.w = np.dot((mean_0 - mean_1), np.linalg.inv(cov))
        self.b = -0.5 * np.dot(np.dot(mean_0, np.linalg.inv(cov)), mean_0) + 0.5 * np.dot(np.dot(mean_1, np.linalg.inv(cov)), mean_1) + np.log(len(class_0) / len(class_1))

    def predict(self,X):
        """
        预测
        :param X: 测试数据
        :return: 预测结果
        """
        y_pred = np.dot(X, self.w) + self.b
        y_pred =  1 / (1 + np.exp(-y_pred))
        return np.clip(y_pred, 1e-8, 1 - 1e-8)



