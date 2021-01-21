import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import ExtraTreeClassifier
from sklearn.metrics import hamming_loss
from utils.generate_stacking_csv import generate_stacking_csv

use_stacking = True

csv_file = r'D:\dataset\2021智慧農業數位分身創新應用競賽\dataset\inputs_and_target'
save_path = r'D:/dataset/2021智慧農業數位分身創新應用競賽/dataset/stacking/'
model_name = r'ExtraTreeClassifier'
train_x = pd.read_csv(csv_file + r'\train_x.csv', index_col=0)
val_x = pd.read_csv(csv_file + r'\val_x.csv', index_col=0)
test_x = pd.read_csv(csv_file + r'\test_x.csv', index_col=0)
train_y = pd.read_csv(csv_file + r'\train_y.csv', index_col=0)
val_y = pd.read_csv(csv_file + r'\val_y.csv', index_col=0)

# 標準化
ss = StandardScaler()
train_x = ss.fit_transform(train_x)
val_x = ss.transform(val_x)
test_x = ss.transform(test_x)
train_y = train_y.values
val_y = val_y.values

# 設定參數
model = ExtraTreeClassifier()

if use_stacking:
    generate_stacking_csv(train_x, val_x, test_x, train_y, val_y, save_path,
                          model_name, model)
else:
    # 訓練模型
    model.fit(train_x, train_y)
    # 將驗證集丟入模型中進行預測
    val_y_pred = model.predict(val_x)
    # 將測試集丟入模型中進行預測
    test_y_pred = model.predict(test_x)
    # 輸出模型評估指標
    print('hamming_loss:', round(hamming_loss(val_y, val_y_pred), 4))
