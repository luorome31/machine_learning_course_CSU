import csv
import train_model

import numpy as np
import matplotlib.pyplot as plt

def plot_mse(mse_array, title):
    """
    可视化MSE值随迭代次数的变化
    :param mse_array: 每次迭代的MSE值
    :param title: 图表标题
    """


    plt.figure(figsize=(10, 6))
    plt.plot(mse_array[10::100], label='MSE')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

def plot_evaluation(predictions, actual, title):
    """
    可视化预测结果与实际值的对比
    """
    plt.figure(figsize=(10, 6))
    plt.plot(predictions, label='Predictions')
    plt.plot(actual, label='Actual Values')
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('PM2.5 Value')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # 进行训练
    X_data, Y_data = train_model.get_train_data()
    model = train_model.LinearRegression(learning_rate=0.1, iterations=100000)
    # 划分训练集和验证集
    X_train, Y_train, X_val, Y_val = train_model.split_train_data(X_data, Y_data)
    print("\nTraining with Gradient Descent:")
    model.gradient_descent(X_train, Y_train)
    model.save_weights('weights_gd.npy')


    print("Training with Adagrad:")
    mse_array= model.adagrad_training_using_validation(X_train, Y_train, X_val, Y_val)
    model.save_weights('weights_adagrad.npy')
    plot_mse(mse_array, 'Adagrad Training MSE')


    # 加载测试数据并预测
    X_test = train_model.get_test_x()

    model.load_weights('weights_gd.npy')
    predictions_gd = model.predict(X_test)

    model.load_weights('weights_adagrad.npy')
    predictions_adagrad = model.predict(X_test)




    # 写入预测结果到CSV文件
    with open('result_gd.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "value"])
        for i, value in enumerate(predictions_gd):
            writer.writerow([f'id_{i}', value])

    with open('result_adagrad.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "value"])
        for i, value in enumerate(predictions_adagrad):
            writer.writerow([f'id_{i}', value])

