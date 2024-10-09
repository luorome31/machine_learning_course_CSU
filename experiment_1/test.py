import csv
import train_model

if __name__ == "__main__":
    # 进行训练
    X_train, Y_train = train_model.get_train_data()
    model = train_model.LinearRegression(learning_rate=0.1, iterations=10000)

    print("\nTraining with Gradient Descent:")
    model.gradient_descent(X_train, Y_train)
    model.save_weights('weights_gd.npy')

    print("Training with Adagrad:")
    model.adagrad(X_train, Y_train)
    model.save_weights('weights_adagrad.npy')

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

