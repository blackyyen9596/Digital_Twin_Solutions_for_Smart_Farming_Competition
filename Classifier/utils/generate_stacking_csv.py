import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, hamming_loss
from tqdm import tqdm


def generate_stacking_csv(train_x, val_x, test_x, train_y, val_y, save_path,
                          model_name, model):
    k = 100
    kf = KFold(n_splits=k, shuffle=False)
    fold = 0
    train_list = []
    val_dic, test_dic = {}, {}
    xgb_train, xgb_val, xgb_test = {}, {}, {}
    with tqdm(kf.get_n_splits(train_x)) as pbar:
        for train_index, test_index in kf.split(train_x):
            # print('train_index:%s , test_index: %s ' % (train_index, test_index))
            inputs = train_x[train_index]
            target = train_y[train_index]
            # 訓練模型
            model.fit(inputs, target)
            # 生成新訓練集
            train_out = model.predict(train_x[test_index])
            for i in range(len(train_out)):
                train_list.append(train_out[i])
            # 生成新驗證集與新測試集
            fold += 1
            val_out = model.predict(val_x)
            val_dic[model_name + str(fold)] = val_out
            test_out = model.predict(test_x)
            test_dic[model_name + str(fold)] = test_out
            pbar.update(1)
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
    print('f1_score:', round(f1_score(val_y, xgb_val, average="macro"), 4))
    print('hamming_loss:', round(hamming_loss(val_y, xgb_val), 4))
