import os
import random

import torch
from torch import optim, nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import *
import torchvision
from model.Transformer import ViT
from Core_Dataset import CoreDataset
from model.CaiT import CaiT
from model.Core_Transformer import Parallel_Transformer, Core_Vision_Transformer
import numpy as np
import pandas as pd

batch_size = 64
epochs = 50
lr = 3e-5
gamma = 0.7
seed = 839

"""

modified at 2023/3/2
 cait输入是224 vit输入是224

"""


def initial_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


initial_seed(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomCrop((256, 256)),
    # transforms.RandomRotation(90),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# test_transforms = transforms.Compose([
#     transforms.Resize((256, 256)),
#     # transforms.RandomCrop((256, 256)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor()
# ])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomCrop((256, 256)),
    # transforms.RandomRotation(90),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = CoreDataset('model_input/train.txt', transforms=train_transforms)
# test_dataset = CoreDataset('model_input/test.txt', transforms=test_transforms)
validation_dataset = CoreDataset('model_input/val.txt', transforms=val_transforms)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

# test_loader = DataLoader(dataset=test_dataset,
#                          batch_size=batch_size,
#                          shuffle=True)

val_loader = DataLoader(dataset=validation_dataset,
                        batch_size=batch_size,
                        shuffle=True)

# model = Base_ViT(image_size=224, patch_size=16, num_classes=4, dim=1024, depth=6, heads=8,
#                  mlp_dim=2048, dropout=0.2, emb_dropout=0.2).to(device)
#
# model = CaiT(image_size=224, patch_size=16, num_classes=3, dim=1024, depth=6, cls_depth=2, heads=8,
#              mlp_dim=2048, dropout=0.1, emb_dropout=0.1).to(device)

# model = torchvision.models.resnet18(pretrained=True)
# num_features = model.fc.in_features
# model.fc = nn.Linear(num_features, 4)
# model = model.to(device)

# model = Core_Vision_Transformer(image_size=224, patch_size=16, num_classes=4, dim=1024, parallel_depth=6, cls_depth=2, heads=8,
#               mlp_dim=2048, dropout=0.1, emb_dropout=0.1).to(device)

model = ViT(image_size=224, patch_size=16, num_classes=3, dim=1024, depth=6, heads=8,
              mlp_dim=2048, dropout=0.1, emb_dropout=0.1).to(device)
# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

# define result list
train_acc_list = []
train_loss_list = []
val_acc_list = []
val_loss_list = []
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for loader, image_data in enumerate(train_loader):
        data, label = image_data
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    # save train result
    print(f"Epoch : {epoch + 1}")
    print(f"train_acc: {epoch_accuracy:.4f}")
    print(f"train_loss: {epoch_loss:.4f}")
    train_acc_list.append(epoch_accuracy.item())
    train_loss_list.append(epoch_loss.item())

    for idx, loader in enumerate([val_loader]):
        # validation or test
        # idx == 0 ---> test_loader
        # idx == 1 ->> val_loader
        with torch.no_grad():
            epoch_accuracy = 0
            epoch_loss = 0
            for data, label in loader:
                data = data.to(device)
                label = label.to(device)

                output = model(data)
                loss = criterion(output, label)

                acc = (output.argmax(dim=1) == label).float().mean()
                epoch_accuracy += acc / len(loader)
                epoch_loss += loss / len(loader)

            # save test and validation result

            print(f"val_acc: {epoch_accuracy:.4f}")
            print(f"val_loss: {epoch_loss:.4f}")
            val_acc_list.append(epoch_accuracy.item())
            val_loss_list.append(epoch_loss.item())

# save result
print(train_acc_list)
print(train_loss_list)
print(val_acc_list)
print(val_loss_list)
result_dataframe = pd.DataFrame({
    'Epoch': [idx + 1 for idx in range(len(train_acc_list))],
    'train_acc': train_acc_list,
    'train_loss': train_loss_list,
    'val_acc': val_acc_list,
    'val_loss': val_loss_list,
})

if not os.path.exists('result'):
    os.mkdir('result')

result_dataframe.to_excel('result/VIT_index_3-2.xlsx', header=True, index=False, encoding='utf-8')

torch.save(model, 'VIT_3-2.pth')
