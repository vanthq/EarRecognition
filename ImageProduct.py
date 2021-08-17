from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from torchvision import datasets, transforms
import torchvision.transforms
from torch.autograd import  Variable
import numpy as np
import matplotlib.pyplot as plt


# normalize = transforms.Normalize((0.1307, ), (0.3081, ))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# train_transformer_ImageNet = transforms.Compose([
#     # transforms.ToPILImage(),
#     transforms.Resize(256),
#     transforms.Lambda(lambda x: x.repeat(3,1,1)),
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     # transforms.Grayscale(num_output_channels=1),
#     normalize
# ])

train_transformer_ImageNet =transforms.Compose([
     transforms.ToTensor(),
     transforms.Resize(256),
    transforms.RandomResizedCrop(224),
     transforms.Lambda(lambda x: x.repeat(3,1,1)),
     normalize
 ])   # 修改的位置



val_transformer_ImageNet = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    normalize
])

class MyDataset(Dataset):
    def __init__(self, filenames, labels, transform):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # image = Image.open(self.filenames[idx]).convert('L')
        image = Image.open(self.filenames[idx])

        image = self.transform(image)
        return image, self.labels[idx]


def split_Train_Val_Data(data_dir, ratio):
    """ the sum of ratio must equal to 1"""
    dataset = ImageFolder(data_dir)  # data_dir精确到分类目录的上一级
    character = [[] for i in range(len(dataset.classes))]
    # print(dataset.class_to_idx)
    for x, y in dataset.samples:  # 将数据按类标存放
        character[y].append(x)
    # print(dataset.samples)

    train_inputs, val_inputs, test_inputs = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    for i, data in enumerate(character):  # data为一类图片
        num_sample_train = int(len(data) * ratio[0])
        # print(num_sample_train)
        num_sample_val = int(len(data) * ratio[1])
        num_val_index = num_sample_train + num_sample_val+1 #➕1是为了拿到所有数据
        for x in data[:num_sample_train]:
            train_inputs.append(str(x))
            train_labels.append(i)
        for x in data[num_sample_train:num_val_index]:
            val_inputs.append(str(x))
            val_labels.append(i)
    # print(len(train_inputs))
    train_dataloader = DataLoader(MyDataset(train_inputs, train_labels, train_transformer_ImageNet),
                                  batch_size=8, shuffle=True)
    val_dataloader = DataLoader(MyDataset(val_inputs, val_labels, val_transformer_ImageNet),
                                batch_size=8, shuffle=False)

    return train_dataloader, val_dataloader

