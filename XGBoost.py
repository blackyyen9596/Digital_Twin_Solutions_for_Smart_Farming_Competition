import os
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, hamming_loss
from sklearn.decomposition import PCA
args = {
    'verbosity': 0,
    'tree_method': 'gpu_hist',
    'eval_metric': 'logloss',
    'use_label_encoder': False,
}
clf_multilabel = OneVsRestClassifier(XGBClassifier(**args))

csv_file = r'D:\dataset\2021智慧農業數位分身創新應用競賽\stacking\xgboost'
label_file = r'D:\dataset\2021智慧農業數位分身創新應用競賽\data_and_label'
train_data = pd.read_csv(
    os.path.join(csv_file, 'train.csv'),
    index_col=0,
)
val_data = pd.read_csv(
    os.path.join(csv_file, 'val.csv'),
    index_col=0,
)
test_data = pd.read_csv(
    os.path.join(csv_file, 'test.csv'),
    index_col=0,
)
train_label = pd.read_csv(
    os.path.join(label_file, 'train_label.csv'),
    index_col=0,
)
val_label = pd.read_csv(
    os.path.join(label_file, 'val_label.csv'),
    index_col=0,
)

pca = PCA(n_components=0.99)
train_data = pca.fit_transform(train_data)
val_data = pca.transform(val_data)
test_data = pca.transform(test_data)

clf_multilabel.fit(train_data, train_label)
val_pred = clf_multilabel.predict(val_data)
test_pres = clf_multilabel.predict(test_data)
test_pres = pd.DataFrame(test_pres)
test_pres.columns = [
    'actuator01', 'actuator02', 'actuator03', 'actuator04', 'actuator05',
    'actuator06', 'actuator07', 'actuator08', 'actuator09', 'actuator10',
    'actuator11'
]
test_pres.to_csv(r'D:\dataset\2021智慧農業數位分身創新應用競賽\results\result_012221.csv')

# 輸出模型評估指標
print('f1_score:', round(f1_score(val_label, val_pred, average="macro"), 4))
print('hamming_loss:', round(hamming_loss(val_label, val_pred), 4))