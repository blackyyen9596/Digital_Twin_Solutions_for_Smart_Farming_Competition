import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models.rnn import rnn
from utils.dataloader_bg import DataLoaderX
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import time
from torch.optim import lr_scheduler
from sklearn.metrics import mean_absolute_error

lr = 1e-2
epochs = 2000

device = torch.device("cuda")

weights_path = r'D:\Github\BlackyYen\BlackyYen-public\Prediction_for_Lily_Price_and_Volume\logs\pre\epoch1000_train_loss_56405.3672_val_loss_27585.7715_mae_107.6021.pth'
# 讀取csv檔案
lily_name = r'FS929'
train_data = pd.read_csv(os.path.join(r'D:\dataset\香水百合價量預測\train_data',
                                      lily_name + '.csv'),
                         index_col=0)
train_label = pd.read_csv(os.path.join(r'D:\dataset\香水百合價量預測\train_label',
                                       lily_name + '.csv'),
                          index_col=0)
val_data = pd.read_csv(os.path.join(r'D:\dataset\香水百合價量預測\val_data',
                                    lily_name + '.csv'),
                       index_col=0)
val_label = pd.read_csv(os.path.join(r'D:\dataset\香水百合價量預測\val_label',
                                     lily_name + '.csv'),
                        index_col=0)

train_data = train_data
train_label = train_label.values
val_data = val_data
val_label = val_label.values

# 標準化
ss = StandardScaler()
train_data = ss.fit_transform(train_data)
val_data = ss.transform(val_data)

train_data = torch.tensor(train_data, dtype=torch.float)
train_label = torch.tensor(train_label, dtype=torch.float)
val_data = torch.tensor(val_data, dtype=torch.float)
val_label = torch.tensor(val_label, dtype=torch.float)

# 切割訓練集
train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
train_loader = DataLoaderX(train_dataset, train_data.shape[0], shuffle=True)
val_dataset = torch.utils.data.TensorDataset(val_data, val_label)
val_loader = DataLoaderX(val_dataset, val_data.shape[0], shuffle=False)

input_size = 40
output_size = 4
model = RNN(input_size=input_size, output_size=output_size)
model.to(device, non_blocking=True)
model.load_state_dict(torch.load(weights_path))
model
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                           T_max=epochs,
                                           eta_min=1e-4,
                                           last_epoch=-1)
criterion = nn.MSELoss()

train_losses, val_losses, mae_list, lr = [], [], [], []

for epoch in range(1, epochs + 1):
    train_loss = 0.0
    val_loss = 0.0
    mae = 0.0
    print('running epoch: {}'.format(epoch))
    # 訓練模式
    start_time = time.time()
    model.train()
    with tqdm(total=len(train_loader),
              desc='train',
              postfix=dict,
              mininterval=0.3) as pbar:
        for inputs, target in train_loader:
            inputs, target = inputs.to(device, non_blocking=True), target.to(
                device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(torch.unsqueeze(inputs, dim=0))
            loss = criterion(torch.squeeze(outputs), target)
            loss.backward()
            optimizer.step()
            loss = loss.detach().cpu().numpy()
            train_loss += loss.item() * inputs.size(0)
            waste_time = time.time() - start_time
            pbar.set_postfix(
                **{
                    'loss': loss.item(),
                    'lr': round(scheduler.get_last_lr()[0], 6),
                    'step/s': waste_time
                })
            pbar.update(1)
            start_time = time.time()
        # scheduler.step()
    lr.append(optimizer.state_dict()['param_groups'][0]['lr'])
    # 評估模式
    start_time = time.time()
    model.eval()
    with tqdm(total=len(val_loader), desc='val', postfix=dict,
              mininterval=0.3) as pbar:
        with torch.no_grad():
            val_correct = 0.0
            for inputs, target in val_loader:
                inputs, target = inputs.to(device), target.to(device)
                outputs = model(torch.unsqueeze(inputs, dim=0))
                loss = criterion(torch.squeeze(outputs), target)
                loss = loss.detach().cpu().numpy()
                val_loss += loss.item() * inputs.size(0)
                mae += mean_absolute_error(torch.squeeze(outputs.cpu()),
                                           target.cpu()) * inputs.size(0)
                waste_time = time.time() - start_time
                pbar.set_postfix(**{'loss': loss.item(), 'step/s': waste_time})
                pbar.update(1)
                start_time = time.time()

    train_loss = train_loss / len(train_loader.dataset)
    val_loss = val_loss / len(val_loader.dataset)
    mae = mae / len(val_loader.dataset)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    mae_list.append(mae)

    print('train loss: {:.6f} \tval loss: {:.6f} \tmae: {:.6f}'.format(
        train_loss, val_loss, mae))

    save_weights_path = './logs'
    save_weight_path = os.path.join(save_weights_path, lily_name)
    if not os.path.isdir(save_weights_path):
        os.mkdir(save_weights_path)
    if not os.path.isdir(save_weight_path):
        os.mkdir(save_weight_path)
    torch.save(
        model.state_dict(),
        './logs/%s/epoch%d_train_loss_%.4f_val_loss_%.4f_mae_%.4f.pth' %
        (lily_name, epoch, train_loss, val_loss, mae))

# 繪製圖
save_images_path = os.path.join('./images', lily_name)
if not os.path.isdir(save_images_path):
    os.mkdir(save_images_path)

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend(loc='best')
plt.savefig(os.path.join(save_images_path, 'loss.jpg'))
plt.show()

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.plot(mae_list, label='Mean Absolute Error')
plt.legend(loc='best')
plt.savefig(os.path.join(save_images_path, 'mae.jpg'))
plt.show()

plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Learning rate')
plt.plot(lr, label='Learning Rate')
plt.legend(loc='best')
plt.savefig(os.path.join(save_images_path, 'lr.jpg'))
plt.show()