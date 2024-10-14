import csv
from collections import defaultdict

import numpy as np



def process_data(file_name):
    if file_name == 'train.csv':
        return process_train_data()
    else:
        return process_test_data()

def process_train_data():
    # 读取CSV文件
    with open("train.csv", mode='r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        data = list(reader)
        y_train = [row[-1] for row in data]
        data = [row[0:-1] for row in data]

    y_train = [0 if str(label) == ' <=50K' else 1 for label in y_train]

    # 根据check_need_one_hot_encoding函数的返回值判断是否需要进行独热编码
    need_encoding = check_need_one_hot_encoding(data)
    categorical_columns = [i for i, need in enumerate(need_encoding) if need]
    continuous_columns = [i for i, need in enumerate(need_encoding) if not need]

    # 提取非连续值列的唯一值
    unique_values = defaultdict(set)
    for row in data:
        for col in categorical_columns:
            unique_values[col].add(row[col])

    # 创建独热编码映射
    one_hot_mapping = {}
    for col, values in unique_values.items():
        one_hot_mapping[col] = {val: idx for idx, val in enumerate(values)}

    # 进行独热编码
    processed_data = []
    for row in data:
        new_row = []
        for i, value in enumerate(row):
            if i in continuous_columns:
                if str(value).strip()=="Male":
                    new_row.append(1)
                elif str(value).strip()=="Female":
                    new_row.append(0)
                else:
                    new_row.append(int(value))
            elif i in categorical_columns:
                one_hot_encoded = [0] * len(one_hot_mapping[i])
                one_hot_encoded[one_hot_mapping[i][value]] = 1
                new_row.extend(one_hot_encoded)
        processed_data.append(new_row)

    return np.array(processed_data), np.array(y_train)


def process_test_data():
    pass


def check_need_one_hot_encoding(data):
    first_row = data[0]
    need_encoding = []
    for item in first_row:
        item = item.strip()
        if item.isdigit():
            need_encoding.append(False)
        elif item not in ['Male', 'Female']:
            need_encoding.append(True)
        else:
            need_encoding.append(False)
    return need_encoding



if __name__ == '__main__':
    X_train , y_train = process_data('train.csv')

    print('done')