import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np

from utils_cls.tools import resize


class load_image_dataset(Dataset):
    def __init__(self, imgs_path, transform=None, img_size=224):
        super(load_image_dataset, self).__init__()
        self.imgs_path = imgs_path
        self.img_files = os.listdir(self.imgs_path)
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, item):
        img = Image.open(self.imgs_path + self.img_files[item])
        #   将图片无变形缩放到224x224
        # img = resize(img, self.img_size)
        img = img.resize((self.img_size, self.img_size))
        label = torch.tensor(int(self.img_files[item].split('_')[0]))
        if self.transform:
            if np.array(img).shape[-1] != 3:
                img = img.convert("RGB")
            img = self.transform(img)

        return img, label
