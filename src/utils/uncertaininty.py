import os
import argparse
import torch
import torch.nn as nn
from dataloader import NucleiDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
from PIL import ImageChops, Image
from  scipy import ndimage
import numpy as np
import matplotlib
from matplotlib import ticker
from cluster import Cluster

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='datasets', type=str, help='dataset root path')
    parser.add_argument('--dataset', default='PanNuke', type=str, help='Mammogram view')

    parser.add_argument('--num_workers', default=5, type=int, help='Number of workers')
    parser.add_argument('--model_save_path', default='models/pannuke/pannuke_pspnet.pth', type=str,
                       help='path to save the model')  # change here
    config = parser.parse_args()
    return config


config = vars(parse_args())
print(config)

device = 'cuda'

test_dataset = NucleiDataset(path=config['path'], dataset=config['dataset'], split='test')
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size =1, num_workers=config['num_workers'])

# load best saved checkpoint
model = torch.load(config['model_save_path'])
model = nn.DataParallel(model.module)




def uncertaininty_map(image, gt_mask, model, n_samples):
    gt_mask = gt_mask[0].cpu().numpy().transpose(1, 2, 0)
    gt_mask = gt_mask[:, :, 0]

    sum = 0
    for i in range(n_samples):
        pred = model.module.predict(image)
        pred = pred[0].cpu().numpy().transpose(1, 2, 0)
        pred = pred[:, :, 0]

        diff = pred - gt_mask
        sum = sum + diff**2

    return np.sqrt(sum/n_samples)




def postprocess_uncertainty(image, h, w):
        """
        Resize uncertainty map. This is strictly for visualisation purposes.
        The output of this function will not be used for anything other
        than visualisation.
        """
        image = Image.fromarray(image)
        image = image.resize((w, h), resample=Image.BICUBIC)
        return np.array(image)


for i, (image, gt_mask) in enumerate(test_dataloader):
    #print(i, image.shape)
    image = image.cuda()

    pred_mask = model.module.predict(image)
    pred_mask = pred_mask[0].cpu().detach().numpy().transpose(1, 2, 0)
    pred_mask = pred_mask[:, :, 0]

    result = uncertaininty_map(image, gt_mask, model, 50)
    print(result.shape)

    #map = postprocess_uncertainty(result, 256, 256)

    image = image[0].cpu().numpy().transpose(1, 2, 0)
    gt_mask = gt_mask[0].cpu().numpy().transpose(1, 2, 0)
    gt_mask = gt_mask[:, :, 0]

    fig, ax_arr = plt.subplots(1, 4, figsize=(16, 5))
    #fig.suptitle('Uncertaininy', fontsize=16)
    ax1, ax2, ax3, ax4  = ax_arr.ravel()

    cmap = matplotlib.cm.jet
    cmap.set_bad(color='k')

    ax1.imshow(image)
    ax1.set_title('Original image', fontsize=16)
    ax1.axis('off')

    ax2.imshow(gt_mask, cmap=cmap)
    ax2.set_title('Ground truth', fontsize=16)
    ax2.axis('off')

    ax3.imshow(pred_mask, cmap=cmap)
    ax3.set_title('Prediction', fontsize=16)
    ax3.axis('off')

    divider = make_axes_locatable(ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax4.matshow(result, cmap=cmap)
    ax4.set_title('Uncertaininty', fontsize=16)
    ax4.axis('off')

    cbar = plt.colorbar(im, cax=cax, orientation='vertical')
    cbar.ax.text(2, 0, 'Blue: Less uncertaininty', fontsize=14)



    plt.tight_layout()
    plt.show()
    #plt.savefig(os.path.join('pannuke_uncertainity', '{}.jpg'.format(i)))

    if i==1:break


