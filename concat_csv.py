import numpy as np
import pandas as pd
import os

sets = ['train', 'val', 'test']

if not os.path.isdir('./stacking/xgboost'):
    os.mkdir('./stacking/xgboost')

for set in sets:
    file_path = os.path.join(r'.\stacking', set)
    file_name = os.listdir(file_path)

    df1 = pd.read_csv(os.path.join(file_path, file_name[0]), index_col=0)
    df2 = pd.read_csv(os.path.join(file_path, file_name[1]), index_col=0)
    df3 = pd.read_csv(os.path.join(file_path, file_name[2]), index_col=0)
    df4 = pd.read_csv(os.path.join(file_path, file_name[3]), index_col=0)
    df5 = pd.read_csv(os.path.join(file_path, file_name[4]), index_col=0)
    df6 = pd.read_csv(os.path.join(file_path, file_name[5]), index_col=0)
    df7 = pd.read_csv(os.path.join(file_path, file_name[6]), index_col=0)
    df8 = pd.read_csv(os.path.join(file_path, file_name[7]), index_col=0)

    data = pd.concat((df1, df2, df3, df4, df5, df6, df7, df8), axis=1)
    data.to_csv(os.path.join(r'.\stacking\xgboost', str(set) + '.csv'))
