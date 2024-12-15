# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 10:23:11 2020

@author: rajgudhe
"""

import os
import glob
import numpy as np
from natsort import natsorted
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class NucleiDataset(Dataset):
    def __init__(self, path, dataset):
        self.path = path
        self.dataset = dataset
               
        self.images = natsorted(glob.glob(os.path.join(self.path, self.dataset, 'images/*')))
        self.masks = natsorted(glob.glob(os.path.join(self.path, self.dataset,  'seg_masks/*')))

        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        self.to_tensor = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((256, 256)),
                                             transforms.ToTensor(),])
        
        self.image_org = cv2.imread(self.images[index], 1)
        self.image = self.to_tensor(self.image_org)
        
        self.mask = cv2.imread(self.masks[index], 0)
        self.mask = (self.mask ==5).astype(np.uint8)*255
        self.mask = self.to_tensor(self.mask)
          
        return self.image, self.mask


class NucleiDatasetTest(Dataset):
    def __init__(self, path, dataset, split):
        self.path = path
        self.dataset = dataset
        self.split = split

        self.images = natsorted(glob.glob(os.path.join(self.path, self.dataset, self.split, 'images/*')))
        self.masks = natsorted(glob.glob(os.path.join(self.path, self.dataset, self.split, 'seg_masks/*')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        self.img_id = os.path.splitext(os.path.split(self.images[index])[-1])[0]

        self.to_tensor = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((256, 256)),
                                             transforms.ToTensor(), ])

        self.image_org = cv2.imread(self.images[index], 1)
        self.image = self.to_tensor(self.image_org)

        self.mask = cv2.imread(self.masks[index], 0)
        self.mask = self.to_tensor(self.mask)

        return self.img_id, self.image, self.mask


      
