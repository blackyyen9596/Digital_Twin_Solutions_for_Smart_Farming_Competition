import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer

train_x = pd.read_csv(r'D:\dataset\2021智慧農業數位分身創新應用競賽\dataset\train_x.csv')
val_x = pd.read_csv(r'D:\dataset\2021智慧農業數位分身創新應用競賽\dataset\val_x.csv')
test_x = pd.read_csv(r'D:\dataset\2021智慧農業數位分身創新應用競賽\dataset\test_x.csv')
train_y = pd.read_csv(r'D:\dataset\2021智慧農業數位分身創新應用競賽\dataset\train_y.csv')
val_y = pd.read_csv(r'D:\dataset\2021智慧農業數位分身創新應用競賽\dataset\val_y.csv')

# 標準化
ss = StandardScaler()
train_x = ss.fit_transform(train_x)
val_x = ss.transform(val_x)
test_x = ss.transform(test_x)

# 設定參數
model = ExtraTreesClassifier()
# 訓練模型
model.fit(train_x, train_y)
# 將驗證集丟入模型中進行預測
val_y_pred = model.predict(val_x)
# 將測試集丟入模型中進行預測
test_y_pred = model.predict(test_x)
# 輸出模型評估指標
print('hamming_loss', round(hamming_loss(val_y, val_y_pred), 4))
