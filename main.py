import torch.cuda
from myDataset import myDataset, visualize
from segnet_model import SegNet
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import albumentations as A
from utils import train, DiceLoss, BinaryDiceLoss


def get_train_augmentation():
    train_transform = [
        # A.HorizontalFlip(p=0.5),
        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        A.Resize(height=224, width=224, always_apply=True),
        # A.Resize(height=224, width=224, always_apply=True, p=1),
    ]
    return A.Compose(train_transform)

def get_validation_augmentation():
    test_transform = [
        A.PadIfNeeded(320, 320, always_apply=True, border_mode=0),
        A.Resize(height=224, width=224, always_apply=True, p=1),
    ]
    return A.Compose(test_transform)


root_dir=r"data/voc2011/VOCdevkit/VOC2012"
train_name_list_path = os.path.join(root_dir, r'ImageSets/Segmentation/train.txt')
test_name_list_path = os.path.join(root_dir, r'ImageSets/Segmentation/val.txt')
with open(train_name_list_path, "r") as f:
    str = f.read()
train_namelist = str.split('\n')
train_namelist = [name for name in train_namelist if name.strip()]

with open(test_name_list_path, "r") as f:
    str = f.read()
test_namelist = str.split('\n')
test_namelist = [name for name in test_namelist if name.strip()]

img_path = os.path.join(os.path.join(root_dir, 'JPEGImages'))
mask_path = os.path.join(os.path.join(root_dir, 'SegmentationClass'))

train_dataset = myDataset(train_namelist, img_path, mask_path, transform=get_train_augmentation())
test_dataset = myDataset(test_namelist, img_path, mask_path, transform=get_validation_augmentation())

trainloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
testloader = DataLoader(dataset=test_dataset, batch_size=32)


device = "cuda" if torch.cuda.is_available() else "cpu"

model = SegNet(3, 21).to(device)
# criterion = DiceLoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs_num = 200

model.load_state_dict(torch.load(r'vgg16_bn-6c64b313.pth'), strict=False)


save_path = 'save_model/SegNet'
train(trainloader, testloader, model, epochs_num, criterion, optimizer, device, save_path)