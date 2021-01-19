import pandas as pd

# 讀取資料集
df = pd.read_csv(r'D:\dataset\2021智慧農業數位分身創新應用競賽\train_data.csv',
                 encoding='utf-8')
# 刪除重複的資料，並保留出現第一次的
df.drop_duplicates(keep='first', inplace=True)
# 將資料打亂
df = df.sample(frac=1.0).reset_index(drop=True)
cut_idx = int(round(0.9 * df.shape[0]))
df_train, df_val = df.iloc[:cut_idx], df.iloc[cut_idx:]
# 輸出train.csv與val.csv檔案
df_train.to_csv(r'D:\dataset\2021智慧農業數位分身創新應用競賽\train.csv', index=False)
df_val.to_csv(r'D:\dataset\2021智慧農業數位分身創新應用競賽\val.csv', index=False)
