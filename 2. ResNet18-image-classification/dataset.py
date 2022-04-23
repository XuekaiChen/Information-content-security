# -*- coding: utf-8 -*-
# Date        : 2022/4/23
# Author      : Chen Xuekai
# Description : Convert image data to a standardized loadable form

import torch
import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, data_path, train=True, transform=None):
        self.data_path = data_path
        self.train_flag = train
        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(size=(224, 224)),
                    transforms.ToTensor(),  # 转为灰度tensor，(Height, Width, Channel)-->(C, H, W)
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalization
                ]
            )
        else:
            self.transform = transform
        self.path_list = os.listdir(data_path)

    def __getitem__(self, idx):
        # img to tensor, label to tensor
        img_path = self.path_list[idx]
        abs_img_path = os.path.join(self.data_path, img_path)
        img = Image.open(abs_img_path)
        img = self.transform(img)

        if self.train_flag is True:
            if img_path.split('.')[0] == 'dog':
                label = 1
            else:
                label = 0
        else:
            label = int(img_path.split('.')[0])
        label = torch.as_tensor(label, dtype=torch.int64)
        return img, label

    def __len__(self) -> int:
        return len(self.path_list)


def dataset_split(full_ds, train_rate, valid_rate):
    train_size = int(len(full_ds) * train_rate)
    valid_size = int(len(full_ds) * valid_rate)
    test_size = len(full_ds) - train_size - valid_size
    not_test_ds, test_ds = torch.utils.data.random_split(full_ds, [train_size+valid_size, test_size])
    train_ds, valid_ds = torch.utils.data.random_split(not_test_ds, [train_size, valid_size])
    return train_ds, valid_ds, test_ds


def dataloader(dataset, batch_size):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, pin_memory=True, num_workers=4)
    return data_loader


if __name__ == '__main__':
    # 模拟测试
    train_path = "data"
    batch_size = 32
    all_data = MyDataset(train_path)

    train_set, valid_set, test_set = dataset_split(all_data, train_rate=0.7, valid_rate=0.1)
    train_loader = dataloader(train_set, batch_size)
    valid_loader = dataloader(valid_set, batch_size)
    test_loader = dataloader(test_set, batch_size)
    print("length of train set: ", len(train_set))
    print("train round: ", len(train_loader))
    print("valid round: ", len(valid_loader))
    print("test round: ", len(test_loader))

    print("testing train data can iterate or not...")
    for item in tqdm(test_loader):
        print("sample size: ", item[0].shape)  # sample:(B,C,H,W)
        print("target size: ", item[1].shape)  # target:(B,)
        print("------------------------------")
        print("first sample: ")
        print(item[0][0])   # 查看标准化处理后的图片
        print("first batch labels")
        print(item[1])
        break
