import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import gc


def preprocessing():
    # 讀取資料集
    train_data = pd.read_csv(
        r'D:\dataset\2021智慧農業數位分身創新應用競賽\dataset\train.csv',
        index_col=0,
        encoding='utf-8',
    )
    val_data = pd.read_csv(
        r'D:\dataset\2021智慧農業數位分身創新應用競賽\dataset\val.csv',
        index_col=0,
        encoding='utf-8',
    )
    test_data = pd.read_csv(
        r'D:\dataset\2021智慧農業數位分身創新應用競賽\org\test_data.csv',
        index_col=0,
        encoding='utf-8',
    )

    y_feature = [
        'actuator01', 'actuator02', 'actuator03', 'actuator04', 'actuator05',
        'actuator06', 'actuator07', 'actuator08', 'actuator09', 'actuator10',
        'actuator11'
    ]
    test_data[y_feature] = -1
    data = pd.concat((train_data, val_data, test_data)).reset_index()
    month_list, time_list = [], []
    for log_time in data['d.log_time']:
        date = log_time.split()[0].split('/')
        month = date[1]
        time = log_time.split()[1]
        month_list.append(month)
        time_list.append(time)
    data_new = {'month': month_list, 'time': time_list}
    data_new = pd.DataFrame(data_new)
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

    train_val = data[data[y_feature[0]] != -1]
    test = data[data[y_feature[0]] == -1]

    # 清理內存
    del data, val_data, test_data
    gc.collect()

    x_features = [i for i in train_val.columns if i not in y_feature]
    index = ['index']
    x_features = list(set(x_features) - set(index))
    # 將資料劃分為訓練集與測試集
    train_x = train_val[x_features][:train_data.shape[0]]
    val_x = train_val[x_features][train_data.shape[0]:]
    test_x = test[x_features]
    train_y = train_val[y_feature][:train_data.shape[0]].astype(int)
    val_y = train_val[y_feature][train_data.shape[0]:].astype(int)
    train_x.to_csv(r'D:\dataset\2021智慧農業數位分身創新應用競賽\dataset\train_x.csv',
                   index=False)
    val_x.to_csv(r'D:\dataset\2021智慧農業數位分身創新應用競賽\dataset\val_x.csv',
                 index=False)
    test_x.to_csv(r'D:\dataset\2021智慧農業數位分身創新應用競賽\dataset\test_x.csv',
                  index=False)
    train_y.to_csv(r'D:\dataset\2021智慧農業數位分身創新應用競賽\dataset\train_y.csv',
                   index=False)
    val_y.to_csv(r'D:\dataset\2021智慧農業數位分身創新應用競賽\dataset\val_y.csv',
                 index=False)
    return train_x, val_x, test_x, train_y, val_y


if __name__ == "__main__":
    preprocessing()