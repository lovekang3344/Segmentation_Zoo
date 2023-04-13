from torchvision import transforms
from torch.utils.data import Dataset
import os
import matplotlib.pylab as plt
import cv2
# 为什么我这里引用了两个图片库呢？就是因为cv和PIL之间的区别
from PIL import Image
import numpy as np
import albumentations as A
import torch


def get_train_augmentation():
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        A.Resize(height=224, width=224, always_apply=True),
        # A.RandomBrightnessContrast(p=0.2),
    ]
    return A.Compose(train_transform)

def get_validation_augmentation():
    test_transform = [
        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        A.Resize(height=224, width=224, always_apply=True),
    ]
    return A.Compose(test_transform)


"""
    初代版本，也是没有发掘什么问题的版本
"""
# @save
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

# @save
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

# @save
def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射"""
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

# @save
def voc_label_indices(colormap, colormap2label):
    """将VOC标签中的RGB值映射到它们的类别索引"""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]


"""
    这次换成PIL，发现规律
"""



class myDataset(Dataset):
    classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
               "train", "tv"]

    def __init__(self, namelist, img_fps, mask_fps, transform=None):
        self.namelist = namelist
        self.img_fps = [os.path.join(img_fps, img_name + ".jpg") for img_name in namelist]
        self.mask_fps = [os.path.join(mask_fps, mask_name + ".png") for mask_name in namelist]
        # 将字符串转为序号量化
        self.class_values = [self.classes.index(cls.lower()) for cls in self.classes]
        # self.colormap2label = voc_colormap2label()

        self.transform = transform
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, idx):

        image = Image.open(self.img_fps[idx]).convert('RGB')
        mask = Image.open(self.mask_fps[idx])
        image, mask = np.array(image), np.array(mask)
        if self.transform is not None:
            sample = self.transform(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]


            image = self.image_transform(image)

            mask[mask > 20] = 0
            # image = self.totensor(image)
            # image = self.Normalize(image)
            # mask = self.totensor(mask)

            mask = torch.from_numpy(np.array(mask)).long()

        return image, mask

def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i+1)
        plt.imshow(image)
        plt.axis('off')
    plt.show()


"""
root_dir="data\\voc2011\\VOCdevkit\\VOC2012"
train_name_list_path = os.path.join(root_dir, 'ImageSets\\Segmentation\\train.txt')
print(train_name_list_path)
with open(train_name_list_path, "r") as f:
    str = f.read()
namelist = str.split('\n')
namelist = [name for name in namelist if name.strip()]
train_img_path = os.path.join(os.path.join(root_dir, 'JPEGImages'))
train_mask_path = os.path.join(os.path.join(root_dir, 'SegmentationClass'))

dataset = myDataset(namelist, train_img_path, train_mask_path, transform=get_train_augmentation())

img, label = dataset[11]
"""

# cv2 读入进来是（H，W, C),展现图片也是需要这样的
# mask = torch.from_numpy(cv2.imread('data/voc2011/VOCdevkit/VOC2012/SegmentationClass/2007_000032.png'))
# img = torch.from_numpy(cv2.imread('data/voc2011/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg'))
# #
# # # y = voc_label_indices(label, voc_colormap2label())
# # # print(y[105:115, 130:140], VOC_CLASSES[1])
# visualize(
#     image=img,
#     mask=mask,
# )

