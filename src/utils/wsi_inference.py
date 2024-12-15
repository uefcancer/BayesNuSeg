import os
import argparse
import torch
import torch.nn as nn
from dataloader import NucleiDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from PIL import ImageChops, Image
Image.MAX_IMAGE_PIXELS = None
import matplotlib
import numpy as np



image = Image.open('WSI/sample1_image.jpg')

to_tensor = transforms.Compose([ transforms.ToTensor() ])

image_tensor = to_tensor(image)

print(image_tensor.shape)
plt.imshow(image)
plt.show()