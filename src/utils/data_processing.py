#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 09:54:25 2021

@author: raju
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import os
import skimage.draw
import numpy as np
from tqdm import tqdm
import cv2
import glob
import warnings
import random

warnings.filterwarnings('ignore')


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def read_nuclei(path):
    if len(path) == 0:
        return None
    img = skimage.io.imread(path)
    
    #input image
    if len(img.shape) > 2:
        img = img[:, :, :3]
    #mask
    else:
        # do nothing
        pass
        
def save_nuclei(path, img):
    "save image"
    skimage.io.imsave(path, img)
    
    
    
label_map = {'Epithelial':1,
             'Lymphocyte':2,
             'Macrophage':4,
             'Neutrophil':3,
            }

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
print(ROOT_DIR)
# Training file directory
IMAGES_FOLDER = os.path.join(ROOT_DIR, "data", "MoNuSAC/training/images_and_annotations/")
MASKS_FOLDER = os.path.join(ROOT_DIR, "data", "MoNuSAC/training/MoNuSAC_masks/")
print(IMAGES_FOLDER, MASKS_FOLDER)


IMAGES_DEST =  os.path.join(ROOT_DIR, "data", "MoNuSAC/training, ""images/")
MASKS_DEST = os.path.join(ROOT_DIR, "data", "MoNuSAC/training", "masks/")

print(IMAGES_DEST)
print(MASKS_DEST)

# Create folders
create_directory(IMAGES_DEST)
create_directory(MASKS_DEST)


IMAGES_SUB_FOLDER = [os.path.join(IMAGES_FOLDER, i) for i in sorted(next(os.walk(IMAGES_FOLDER))[1])]
IMAGES_SUB_FOLDER[:5]


MASKS_SUB_FOLDER = [os.path.join(MASKS_FOLDER, i) for i in sorted(next(os.walk(MASKS_FOLDER))[1])]
MASKS_SUB_FOLDER[:5]


a = sorted(next(os.walk(IMAGES_FOLDER))[1])
b = sorted(next(os.walk(MASKS_FOLDER))[1])
print(a[:2], b[:2])

for x,y in zip(a,b):
    if x == y:
        print(True)
    else:
        print(False)
        
        
        