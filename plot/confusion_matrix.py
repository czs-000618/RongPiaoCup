import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import torch
# from torchvision.transforms import transforms
from torchvision import *
from Core_Dataset import CoreDataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def make_conf_matrix(preds, labels, confusion_matrix):
    preds = torch.argmax(preds, 1)

    for predict, t in zip(preds, labels):
        # print(predict)
        # print(t)
        confusion_matrix[predict, t] += 1
    return confusion_matrix


n_class = 3
# 创建一个空的
confusion_matrix = torch.zeros(n_class, n_class)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.Resize((224, 224)),
    # transforms.Grayscale(num_output_channels=3),
    # transforms.RandomRotation(degrees=90),
    # transforms.RandomCrop(224),
    # vgg专用

    # transforms.RandomCrop(112),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = CoreDataset(r'../model_input/train.txt', transforms=train_transform)
test_dataset = CoreDataset(r'../model_input/val.txt', transforms=train_transform)
train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=32,
                                                shuffle=True,
                                                num_workers=0)

test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=32,
                                               shuffle=True,
                                               num_workers=0)

# predict(train_data_loader)
# 求confusion matrix
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pthfile = r'../VIT_3-2.pth'

model = torch.load(pthfile)
model = model.eval().to(device)
for index, inputs in enumerate(test_data_loader):
    input, labels = inputs
    if torch.cuda.is_available():
        input = input.to(device)
        labels = labels.to(device)

    output = model(input)
    confusion_matrix = make_conf_matrix(output, labels, confusion_matrix)

print(confusion_matrix)

import seaborn as sns

index = ['mudstone', 'limestone', 'algal']
dataframe_confusion_matrix = pd.DataFrame(confusion_matrix.numpy(),
                                          index=index,
                                          columns=index)
plt.figure(figsize=(15, 15))
sns.heatmap(dataframe_confusion_matrix, annot=True, cmap='Blues', fmt='g', annot_kws={"fontsize": 20})

# plt.xticks(rotation=45)
plt.yticks(rotation=360)
plt.yticks(fontproperties="Arial", size=25, weight="normal")
plt.xticks(fontproperties="Arial", size=25, weight="normal")
plt.savefig('VIT_test_confusion_matrix.jpg')
plt.show()
