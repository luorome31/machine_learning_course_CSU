import csv
from collections import defaultdict

import numpy as np

def standardize(data):
    """
    标准化数据
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

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
        data = [row for row in data if row[-2] != ' Holand-Netherlands']
        y_train = [row[-1] for row in data]
        data = [row[:-1] for row in data]

    y_train = [0 if str(label) == ' <=50K' else 1 for label in y_train]

    processed_data = one_hot_encoding(data)
    return np.array(processed_data), np.array(y_train)


def process_test_data():
    with open("test.csv", mode='r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        data = list(reader)

    processed_data = one_hot_encoding(data)
    return np.array(processed_data)

def one_hot_encoding(data):

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
    #
    # # 记录每一列的独热编码长度,并保存到文件
    # count_one_hot_encoding_length(get_name(),one_hot_mapping)
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
                    new_row.append(float(value))
            elif i in categorical_columns:
                one_hot_encoded = [0] * len(one_hot_mapping[i])
                one_hot_encoded[one_hot_mapping[i][value]] = 1
                new_row.extend(one_hot_encoded)
        processed_data.append(new_row)

    return np.array(processed_data)

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

# def count_one_hot_encoding_length(filename,one_hot_mapping):
#     with open(f'{next(filename)}.txt', 'w') as file:
#         for key,value in one_hot_mapping[13].items():
#             file.write(f'{key}:{value}\n')


if __name__ == '__main__':
    X_train , y_train = process_data('train.csv')

    print('done')