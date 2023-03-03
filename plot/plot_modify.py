import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

label_font = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 30,
}

data_path_list = [r'../result/CaiT_index_3-2.xlsx', r'../result/VIT_index_3-2.xlsx']
for idx, data_path in enumerate(data_path_list):
    data = pd.read_excel(data_path)
    epoch = data['Epoch'].tolist()[::4]
    train_acc_list = data['train_acc'].tolist()[::4]
    train_loss_list = data['train_loss'].tolist()[::4]
    val_acc_list = data['val_acc'].tolist()[::4]
    val_loss_list = data['val_loss'].tolist()[::4]

    acc_list = [train_acc_list, val_acc_list]
    loss_list = [train_loss_list, val_loss_list]
    # acc_marker_list = ['o', 'o', '^', '^']
    # color_list = ['blue', 'blue', 'red', 'red']

    plt.figure(figsize=(10, 10), dpi=200)
    plt.plot(epoch, loss_list[1], marker='o', markersize=10, color="blue")
    plt.yticks(fontproperties="Arial", size=20, weight="normal")
    plt.xticks(fontproperties="Arial", size=20, weight="normal")
    plt.xlabel('Epoch', fontdict=label_font)
    plt.ylabel('Loss', fontdict=label_font)
    plt.savefig('Cait_val_loss.jpg' if idx == 0 else 'VIT_val_loss.jpg')
    plt.show()
