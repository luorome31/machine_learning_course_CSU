import csv
import numpy as np


def get_test_x(file_path='test.csv'):
    """
    加载测试数据
    """
    with open(file_path, 'r', encoding='big5') as f:
        reader = csv.reader(f)
        data = list(reader)

    # 处理数据
    data = [row[2:] for row in data]
    data = [[0 if cell == 'NR' else float(cell) for cell in row] for row in data]

    X_test = []
    test_attribute = len(data) / 18
    for group in range(int(test_attribute)):
        X_test.append([1])  # 添加偏置项
        for attribute in range(18):
            X_test[group].extend(data[group * 18 + attribute][:9])

    return np.array(X_test)


def get_train_data(file_path='train.csv'):
    """
    加载训练数据
    """
    with open(file_path) as f:
        reader = csv.reader(f)
        data = list(reader)

    # 处理数据
    data = [row[3:] for row in data[1:]]
    data = [[0 if cell == 'NR' else float(cell) for cell in row] for row in data]

    init_data = [[] for _ in range(18)]
    for i in range(len(data)):
        index = i % 18
        init_data[index].extend(data[i])

    X_train = []
    Y_train = []
    for month in range(12):
        for group in range(471):
            features = []
            for attribute in range(18):
                features.extend(init_data[attribute][480 * month + group:480 * month + group + 9])
            X_train.append(features)
            Y_train.append(init_data[9][480 * month + group + 9])

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    # 添加偏置项
    X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)

    return X_train, Y_train

def split_train_data(X_train, Y_train, ratio=0.8):
    """
    划分训练集和验证集
    """
    num_train = int(X_train.shape[0] * ratio)
    X_train, Y_train, X_val, Y_val = X_train[:num_train], Y_train[:num_train], X_train[num_train:], Y_train[num_train:]
    return X_train, Y_train, X_val, Y_val

class LinearRegression:
    """
    线性回归模型类，支持普通梯度下降和Adagrad优化
    """
    def __init__(self, learning_rate=0.1, iterations=100):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None

    def evaluate_model(self, X_val, Y_val):
        """
        评估模型在验证集上的性能
        :return: 均方误差 (MSE)
        """
        predictions = self.predict(X_val)
        mse = np.mean((predictions - Y_val) ** 2)
        return mse
    def gradient_descent(self, X, y):
        """
        使用普通梯度下降优化
        """
        m, n = X.shape
        self.weights = np.zeros(n)
        for i in range(self.iterations):
            predictions = X.dot(self.weights)
            errors = predictions - y
            gradients = 2 / m * X.T.dot(errors)
            self.weights -= self.learning_rate * gradients
            if i % 10 == 0:
                cost = np.mean(errors ** 2)
                print(f'ordinary_gradient Iteration {i}, Cost: {cost}')


    def adagrad_training_using_validation(self, X_train, Y_train, X_val, Y_val):
        """
        使用Adagrad优化，同时在验证集上评估模型性能
        """
        m, n = X_train.shape
        self.weights = np.zeros(n)
        grad_square = np.zeros(n)
        mse_array = []


        for i in range(self.iterations):
            predictions = X_train.dot(self.weights)
            errors = predictions - Y_train
            gradients = 2 / m * X_train.T.dot(errors)

            # 累加梯度平方
            grad_square += gradients ** 2

            # 计算Adagrad更新
            ada = np.sqrt(grad_square + 1e-8)
            self.weights -= self.learning_rate * gradients / ada

            if i % 10 == 0:
                cost = np.mean(errors ** 2)
                print(f'adagrad Iteration {i}, Cost: {cost}')


            mse_array.append(self.evaluate_model( X_val, Y_val))

        return mse_array

    def adagrad(self, X, y):
        """
        使用Adagrad优化
        """
        m, n = X.shape
        self.weights = np.zeros(n)
        grad_square = np.zeros(n)

        for i in range(self.iterations):
            predictions = X.dot(self.weights)
            errors = predictions - y
            gradients = 2 / m * X.T.dot(errors)

            # 累加梯度平方
            grad_square += gradients ** 2

            # 计算Adagrad更新
            ada = np.sqrt(grad_square + 1e-8)
            self.weights -= self.learning_rate * gradients / ada

            if i % 10 == 0:
                cost = np.mean(errors ** 2)
                print(f'adagrad Iteration {i}, Cost: {cost}')

    def predict(self, X):
        """
        使用模型进行预测
        """
        return np.dot(X, self.weights)

    def save_weights(self, file_name):
        """
        保存权重
        """
        np.save(file_name, self.weights)

    def load_weights(self, file_name):
        """
        加载权重
        """
        self.weights = np.load(file_name)



