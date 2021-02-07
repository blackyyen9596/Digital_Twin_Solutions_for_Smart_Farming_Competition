import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

org_path = r'D:\dataset\2021智慧農業數位分身創新應用競賽\org'
all_path = r'D:\dataset\2021智慧農業數位分身創新應用競賽\org\train-test.csv'
save_path = r'D:\dataset\2021智慧農業數位分身創新應用競賽\dataset_lstm'

train_df = pd.read_csv(os.path.join(org_path, 'train_data.csv'), index_col=0)
col_name = train_df.columns.to_list()

# 在測試資料新增label欄位並填入NAN
test_df = pd.read_csv(os.path.join(org_path, 'test_data.csv'), index_col=0)
col_num = train_df.shape[1]
data_num = test_df.shape[1]
nan_array = np.full([test_df.shape[0], col_num - data_num], np.nan)
test_df = test_df.reindex(columns=col_name)
test_df.iloc[:, data_num:] = nan_array
# concat 訓練資料與測試資料
all_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
# 對時間排列
all_df['d.log_time'] = pd.to_datetime(all_df['d.log_time'])
all_df.sort_values('d.log_time', inplace=True)
all_df.to_csv(all_path, index=None)

#正規化
data = all_df.iloc[:, 1:data_num]
x_scaler = StandardScaler().fit(data)
data = x_scaler.transform(data)
all_df.iloc[:, 1:data_num] = data

# 前10個時間點(包含當下)當做訓練資料，若遇到nan則跳過
n = 288
train_data = []
train_label = []
test_data = []
for i in tqdm(range(len(all_df) - n)):
    if np.isnan(all_df.iloc[i + n - 1]['actuator01']):  # 測試集
        data = all_df.iloc[i:i + n, 1:data_num].values  # 第一行是時間
        test_data.append(data)
    else:  # 訓練驗證集
        data = all_df.iloc[i:i + n, 1:data_num].values  # 第一行是時間
        label = all_df.iloc[i + n - 1, data_num:].values
        train_data.append(data)
        train_label.append(label)

train_data = np.array(train_data)
train_label = np.array(train_label)
test_data = np.array(test_data)
print('train_data:', train_data)
print('train_label:', train_label.shape)
print('test_data:', test_data.shape)

# 存檔.npy
np.save(os.path.join(save_path, 'train_val_data.npy'), train_data)
np.save(os.path.join(save_path, 'train_val_label.npy'), train_label)
np.save(os.path.join(save_path, 'test_data.npy'), test_data)