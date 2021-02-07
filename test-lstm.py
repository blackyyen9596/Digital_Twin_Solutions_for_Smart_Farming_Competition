import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import utils
from torch.utils.data import Dataset, DataLoader
from models import rnn
import math
from tqdm import tqdm
from utils.utils import TestSetloader

PATH_submission_example = r'D:\dataset\2021智慧農業數位分身創新應用競賽\org\submission.csv'
PATH_submission = r'./results/submission.csv'
PATH_test = r'D:\dataset\2021智慧農業數位分身創新應用競賽\dataset_lstm\test_data.npy'

threshold = torch.tensor([0.5])

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Test on GPU...')
else:
    device = torch.device('cpu')

data = np.load(PATH_test, allow_pickle=True)
data = torch.tensor(data, dtype=torch.float)
testset = TestSetloader(data)
testloader = DataLoader(testset, batch_size=len(testset.data), shuffle=False)

# 定義模型
model = rnn.rnn(input_size=18, output_size=11)
model.to(device)

# 載入預訓練權重
model.load_state_dict(
    torch.load('./weights/epoch119-loss0.0835-val_loss0.1468-f10.7070.pth'))

#評估模式
model.eval()
outputs_list = np.empty((0, 11))  # 11維特徵
total_val_loss = 0
with tqdm(testloader) as pbar:
    with torch.no_grad():
        for inputs in testloader:
            inputs = inputs.to(device)
            inputs = inputs.permute(1, 0, 2)
            outputs = model(inputs)
            outputs = (outputs.cpu() > threshold).float() * 1
            outputs_list = np.vstack((outputs_list, outputs))
            #更新進度條
            pbar.set_description('test')
            pbar.update(1)
print(outputs_list.shape)
df = pd.read_csv(PATH_submission_example, index_col=0)
df.iloc[:, :] = outputs_list
df.to_csv(PATH_submission)