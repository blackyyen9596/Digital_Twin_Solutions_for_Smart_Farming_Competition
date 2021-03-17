import torch
import torch.nn as nn
import torch.utils.data
from torch.optim import lr_scheduler
import copy
from models.dnn import dnn
from tqdm import tqdm
import pandas as pd
import numpy as np
from preprocessing import preprocessing as pre

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, hamming_loss
import os

if not os.path.isdir('./stacking'):
    os.mkdir('./stacking')
if not os.path.isdir('./stacking/train'):
    os.mkdir('./stacking/train')
if not os.path.isdir('./stacking/val'):
    os.mkdir('./stacking/val')
if not os.path.isdir('./stacking/test'):
    os.mkdir('./stacking/test')

model_name = 'DNN'
k = 100
epochs = 500
csv_file = r'D:\dataset\2021智慧農業數位分身創新應用競賽\data_and_label'
save_path = r'.\stacking'

train_data = pd.read_csv(csv_file + r'\train_data.csv', index_col=0)
val_data = pd.read_csv(csv_file + r'\val_data.csv', index_col=0)
test_data = pd.read_csv(csv_file + r'\test_data.csv', index_col=0)
train_label = pd.read_csv(csv_file + r'\train_label.csv', index_col=0)
val_label = pd.read_csv(csv_file + r'\val_label.csv', index_col=0)

ss = StandardScaler()
train_data = ss.fit_transform(train_data)
val_data = ss.transform(val_data)
test_data = ss.transform(test_data)
train_label = train_label.values
val_label = val_label.values

train_data = torch.tensor(train_data)
val_data = torch.tensor(val_data)
test_data = torch.tensor(test_data)
train_label = torch.tensor(train_label)
val_label = torch.tensor(val_label)

learning_rate = 1e-3
weight_decay = 0

device = torch.device('cuda')

kf = KFold(n_splits=k, shuffle=False)
fold = 0
switch = True
threshold = [0.5]
train_list = []
val_dic, test_dic, xgb_train, xgb_val, xgb_test = {}, {}, {}, {}, {}

for train_index, test_index in kf.split(train_data):
    inputs = train_data[train_index]
    target = train_label[train_index]
    # 切割訓練集
    train_batch_size = inputs.shape[0]
    val_batch_size = val_data.shape[0]
    train_dataset = torch.utils.data.TensorDataset(inputs, target)
    train_iter = torch.utils.data.DataLoader(train_dataset,
                                             train_batch_size,
                                             shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(val_data, val_label)
    val_iter = torch.utils.data.DataLoader(val_dataset,
                                           val_batch_size,
                                           shuffle=True)
    model = dnn(features=train_data.shape[1])
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss(size_average=None,
                                reduce=None,
                                reduction='mean',
                                beta=1.0)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                               T_max=epochs,
                                               eta_min=1e-5,
                                               last_epoch=-1)
    val_loss = float('inf')
    for epoch in range(epochs):
        print('fold: {} running epoch: {}'.format(fold + 1, epoch + 1))
        # 訓練模型
        model.train()
        with tqdm(train_iter) as pbar:
            for inputs, target in train_iter:
                inputs = inputs.to(torch.float32).to(device)
                target = target.to(torch.float32).to(device)
                optimizer.zero_grad()
                loss = criterion(model(inputs), target)
                loss.backward()
                optimizer.step()
                pbar.update(1)
                pbar.set_postfix(**{'loss': loss.item()})
        scheduler.step()
        # 驗證模型
        model.eval()
        with tqdm(train_iter) as pbar:
            with torch.no_grad():
                for inputs, target in val_iter:
                    inputs = inputs.to(torch.float32).to(device)
                    target = target.to(torch.float32).to(device)
                    val_loss = criterion(model(inputs.float()), target.float())
                    pbar.update(1)
                    pbar.set_postfix(**{
                        'loss': val_loss.item(),
                    })
        # 複製最好的模型參數資料
        if switch == True:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            switch = False
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_wts)
    # 生成新訓練集
    train_out = model(train_data[test_index].to(
        torch.float32).to(device)).cpu().detach().numpy()
    train_out = (train_out > threshold) * 1
    for i in range(len(train_out)):
        train_list.append(train_out[i])
    # 生成新驗證集與新測試集
    fold += 1
    val_out = model(val_data.to(
        torch.float32).to(device)).cpu().detach().numpy()
    train_out = (val_out > threshold) * 1
    val_dic[model_name + str(fold)] = val_out
    test_out = model(test_data.to(
        torch.float32).to(device)).cpu().detach().numpy()
    test_out = (test_out > threshold) * 1
    test_dic[model_name + str(fold)] = test_out

# 生成新的訓練集
list = []
for i in range(np.array(train_list).shape[1]):
    for j in range(np.array(train_list).shape[0]):
        list.append(train_list[j][i])
    xgb_train[model_name + str(i + 1)] = list
    list = []

xgb_train = pd.DataFrame(xgb_train)
xgb_train.to_csv('{}.csv'.format(
    os.path.join(save_path, 'train/' + model_name + '_train')))

# 生成新的驗證集
total_out = 0
val_list = []
list = []
for i in range(fold):
    total_out += val_dic[model_name + str(i + 1)]
for i in range(total_out.shape[0]):
    for j in range(total_out.shape[1]):
        if total_out[i][j] >= int(k // 2):
            list.append(1)
        else:
            list.append(0)
    val_list.append(np.array(list, dtype='int64'))
    list = []

for i in range(np.array(val_list).shape[1]):
    for j in range(np.array(val_list).shape[0]):
        list.append(val_list[j][i])
    xgb_val[model_name + str(i + 1)] = list
    list = []

xgb_val = pd.DataFrame(xgb_val)
xgb_val.to_csv('{}.csv'.format(
    os.path.join(save_path, 'val/' + model_name + '_val')))

# 生成新的測試集
total_out = 0
test_list = []
list = []
for i in range(fold):
    total_out += test_dic[model_name + str(i + 1)]
for i in range(total_out.shape[0]):
    for j in range(total_out.shape[1]):
        if total_out[i][j] >= int(k // 2):
            list.append(1)
        else:
            list.append(0)
    test_list.append(np.array(list, dtype='int64'))
    list = []

for i in range(np.array(test_list).shape[1]):
    for j in range(np.array(test_list).shape[0]):
        list.append(test_list[j][i])
    xgb_test[model_name + str(i + 1)] = list
    list = []

xgb_test = pd.DataFrame(xgb_test)
xgb_test.to_csv('{}.csv'.format(
    os.path.join(save_path, 'test/' + model_name + '_test')))

# 輸出模型評估指標
print(
    'f1_score:',
    round(f1_score(val_label.cpu().detach().numpy(), xgb_val, average="macro"),
          4))
print('hamming_loss:',
      round(hamming_loss(val_label.cpu().detach().numpy(), xgb_val), 4))
