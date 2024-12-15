import os
import argparse
import torch
import torch.nn as nn
from dataloader import NucleiDatasetTest
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

    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers')
    parser.add_argument('--model_save_path', default='models/pannuke/pannuke_unetplus.pth', type=str,
                       help='path to save the model')  # change here
    config = parser.parse_args()
    return config


config = vars(parse_args())
print(config)

device = 'cuda'

test_dataset = NucleiDatasetTest(path=config['path'], dataset=config['dataset'], split='test')
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


for i, (img_id, image, gt_mask) in enumerate(test_dataloader):
    print(i)
    image = image.cuda()

    pred_mask = model.module.predict(image)
    pred_mask = pred_mask[0].cpu().detach().numpy().transpose(1, 2, 0)
    pred_mask = pred_mask[:, :, 0]

    result = uncertaininty_map(image, gt_mask, model, 50)
    print(result.shape)

    map = postprocess_uncertainty(result, 256, 256)

    image = image[0].cpu().numpy().transpose(1, 2, 0)
    gt_mask = gt_mask[0].cpu().numpy().transpose(1, 2, 0)
    gt_mask = gt_mask[:, :, 0]




    fig, ax_arr = plt.subplots(1,2, figsize=(9,4))
    #fig.suptitle('Uncertaininy', fontsize=16)
    ax1, ax2  = ax_arr.ravel()

    cmap = matplotlib.cm.jet
    cmap.set_bad(color='k')

    ax1.imshow(image)
    ax1.contour(gt_mask, colors='red', linewidths=0.4)
    ax1.contour(pred_mask, colors='blue', linewidths=0.4)
    ax1.set_title('Segmentation', fontsize=16, fontweight='bold')
    ax1.axis('off')

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax2.matshow(map, cmap=cmap)
        #x2.matshow(map, cmap=cmap)
    ax2.set_title('Uncertaininty', fontsize=16, fontweight='bold')
    ax2.axis('off')

    cbar = plt.colorbar(im, cax=cax, orientation='vertical')
        #cbar.ax.text(2, 0, 'Blue: Less uncertaininty', fontsize=16)

        # Save just the portion _inside_ the second axis's boundaries
        #extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        #fig.savefig('ax2_figure.png', bbox_inches=extent)



    plt.tight_layout()
        #plt.suptitle(img_id[0])
    #plt.savefig(os.path.join('cryonuseg_uncertaininty', '{}.jpg'.format(i)))
    plt.show()

    if i==4:
        break







