import os
import argparse
import torch
import torch.nn as nn
from dataloader import NucleiDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
import numpy as np



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='datasets', type=str, help='dataset root path')
    parser.add_argument('--dataset', default='NuCLS', type=str, help='Mammogram view')

    parser.add_argument('--num_workers', default=5, type=int, help='Number of workers')
    parser.add_argument('--model_save_path', default='models/nucls_unet.pth', type=str,
                       help='path to save the model')  # change here
    config = parser.parse_args()
    return config


config = vars(parse_args())
print(config)


test_dataset = NucleiDataset(path=config['path'], dataset=config['dataset'], split='test')
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size =1, num_workers=config['num_workers'])

# load best saved checkpoint
model = torch.load(config['model_save_path'])
model = nn.DataParallel(model.module)


for i, (image, gt_mask) in enumerate(test_dataloader):
    print(i, image.shape)
    image = image.cuda()
    pred_mask  = model.module.predict(image)

    image = image[0].cpu().numpy().transpose(1, 2, 0)
    #image = image[:, :, 0]

    gt_mask = gt_mask[0].cpu().numpy().transpose(1,2,0)
    gt_mask = gt_mask[:, :, 0]

    pred_mask = pred_mask[0].cpu().detach().numpy().transpose(1, 2, 0)
    pred_mask = pred_mask[:, :, 0]

    plt.subplot(131)
    plt.imshow(image)
    plt.title('Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(gt_mask, cmap='jet')
    plt.title('Ground truth')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(pred_mask, cmap='jet')
    plt.title('Predictions')
    plt.axis('off')

    plt.show()

    if i==4:
        break



















