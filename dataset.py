from __future__ import print_function, division
import os, torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from utils import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def get_svhn_loaders(transform, config):
    loader_train = DataLoader(SVHN(transform, train=True, tiny=config['tiny_ds']), batch_size=config['bz'])
    loader_test = DataLoader(SVHN(transform, train=False, tiny=config['tiny_ds']), batch_size=config['bz'])
    return loader_train, loader_test

class SVHN(Dataset):

    def __init__(self, input_transform, train=True, device = torch.device('cpu'), tiny=False):
        '''
        Input:
            - Dataset: tuple (X,y).  (N,32,32,3), (N,10)
        '''

        self.X, self.y = load_svhn(train)
        if tiny:
            self.X = self.X[:128]
            self.y = self.y[:128]
        self.input_transform = input_transform
        self.device = device
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        ''' Normalize each image and 1hot encode each label'''
        img = self.X[idx]
        img = img.squeeze().astype('float32')
        
        if self.input_transform is not None: img = self.input_transform(img)
        img = img.to(self.device)
        
        return img