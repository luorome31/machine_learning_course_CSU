import csv
import train_model

import matplotlib.pyplot as plt
import seaborn as sns


def plot_mse(mse_array, title):
    """
    可视化MSE值随迭代次数的变化
    :param mse_array: 每次迭代的MSE值
    :param title: 图表标题
    """
    # 设置美观的主题
    sns.set(style="whitegrid")

    # 创建图表并设置大小
    plt.figure(figsize=(12, 7))

    # 绘制MSE曲线，增加颜色和线型
    plt.plot(mse_array[10::100], color='firebrick', linestyle='-', linewidth=2, label='MSE')

    # 标记起点和终点
    plt.scatter(0, mse_array[10], color='green', zorder=5, label='Start', s=80)  # 起点标记
    plt.scatter(len(mse_array[10::100]) - 1, mse_array[-1], color='blue', zorder=5, label='End', s=80)  # 终点标记

    # 设置标题和标签
    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel('Iteration (Sampled)', fontsize=14)
    plt.ylabel('Mean Squared Error', fontsize=14)

    # 添加图例
    plt.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()  # 自动调整布局
    plt.show()


def plot_evaluation(predictions, actual, title):
    """
    可视化预测结果与实际值的对比
    """
    sns.set(style="whitegrid")

    plt.figure(figsize=(12, 7))

    plt.plot(predictions, label='Predictions', color='royalblue', linestyle='--', linewidth=2)
    plt.plot(actual, label='Actual Values', color='darkorange', linewidth=2)
    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel('Sample Index', fontsize=14)
    plt.ylabel('PM2.5 Value', fontsize=14)
    plt.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 进行训练
    X_data, Y_data = train_model.get_train_data()
    model = train_model.LinearRegression(learning_rate=0.1, iterations=30000)
    # 划分训练集和验证集
    X_train, Y_train, X_val, Y_val = train_model.split_train_data(X_data, Y_data)
    print("\nTraining with Gradient Descent:")
    model.gradient_descent(X_train, Y_train)
    model.save_weights('weights_gd.npy')


    print("Training with Adagrad:")
    mse_array= model.adagrad_training_using_validation(X_train, Y_train, X_val, Y_val)
    model.save_weights('weights_adagrad.npy')
    plot_mse(mse_array, 'Adagrad Validation MSE')

    validation_predictions = model.predict(X_val)
    plot_evaluation(validation_predictions, Y_val, 'Adagrad Validation Predictions')


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

