# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 11:58:07 2017

@author: Biagio Brattoli
"""
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class DataLoader(data.Dataset):
    def __init__(self, root, data_path, label_path):
        self.root = root
        self.data_file = np.load(data_path)
        self.label_file = np.load(label_path)
        self.N = self.label_file.size

        self.__image_transformer = transforms.Compose([
            transforms.Resize(128, Image.BILINEAR),
            transforms.ToTensor()])
       
    def __getitem__(self, index):
        framename = self.root + '/' + self.data_file[index]
        img = Image.open(framename).convert('RGB')
        if img.size[0] != 128:
            img = self.__image_transformer(img)

        return img, self.label_file[index]

    def __len__(self):
        return self.N
