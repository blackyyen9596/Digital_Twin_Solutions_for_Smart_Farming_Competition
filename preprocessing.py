import os
import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocessing():
    # 讀取資料集
    read_path = r'D:\dataset\2021智慧農業數位分身創新應用競賽\dataset'
    save_path = r'D:\dataset\2021智慧農業數位分身創新應用競賽\data_and_label'

    train_data = pd.read_csv(os.path.join(read_path, 'train.csv'), index_col=0)
    val_data = pd.read_csv(os.path.join(read_path, 'val.csv'), index_col=0)
    test_data = pd.read_csv(os.path.join(read_path, 'test.csv'), index_col=0)
    label_feature = [
        'actuator01', 'actuator02', 'actuator03', 'actuator04', 'actuator05',
        'actuator06', 'actuator07', 'actuator08', 'actuator09', 'actuator10',
        'actuator11'
    ]
    test_data[label_feature] = -1
    data = pd.concat((train_data, val_data, test_data)).reset_index(drop=True)
    month_list, time_list = [], []
    for log_time in data['d.log_time'].values:
        date = log_time.split()[0].split('-')
        month = date[1]
        time = log_time.split()[1]
        month_list.append(month)
        time_list.append(time)
    data_new = {'month': month_list, 'time': time_list}
    data_new = pd.DataFrame(data_new).reset_index(drop=True)
    data = pd.concat([data, data_new], axis=1)
    data.drop(columns=['d.log_time'], axis=1, inplace=True)

    # category feature one_hot
    le = LabelEncoder()
    cate_feature = ['month', 'time']
    for item in cate_feature:
        data_new = le.fit_transform(data[item])
        data_new = pd.DataFrame(data_new)
        data_new.columns = [str('le_' + item)]
        data = pd.concat([data, data_new], axis=1)
    data.drop(cate_feature, axis=1, inplace=True)

    train_val = data[data[label_feature[0]] != -1]
    test = data[data[label_feature[0]] == -1]

    # 清理內存
    del data, val_data, test_data
    gc.collect()

    data_feature = [i for i in train_val.columns if i not in label_feature]
    index = ['index']
    data_feature = list(set(data_feature) - set(index))
    # 將資料劃分為訓練集與測試集
    train_data = train_val[data_feature][:train_data.shape[0]].reset_index(
        drop=True)
    val_data = train_val[data_feature][train_data.shape[0]:].reset_index(
        drop=True)
    test_data = test[data_feature].reset_index(drop=True)
    train_label = train_val[label_feature][:train_data.shape[0]].astype(
        int).reset_index(drop=True)
    val_label = train_val[label_feature][train_data.shape[0]:].astype(
        int).reset_index(drop=True)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    train_data.to_csv(os.path.join(save_path, 'train_data.csv'))
    val_data.to_csv(os.path.join(save_path, 'val_data.csv'))
    test_data.to_csv(os.path.join(save_path, 'test_data.csv'))
    train_label.to_csv(os.path.join(save_path, 'train_label.csv'))
    val_label.to_csv(os.path.join(save_path, 'val_label.csv'))


if __name__ == "__main__":
    preprocessing()