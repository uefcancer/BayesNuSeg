import os
import argparse
import torch
import torch.nn as nn
from dataloader import NucleiDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
from PIL import ImageChops, Image
import matplotlib
import numpy as np



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='datasets', type=str, help='dataset root path')
    parser.add_argument('--dataset', default='PanNuke', type=str, help='Mammogram view')

    parser.add_argument('--num_workers', default=5, type=int, help='Number of workers')

    config = parser.parse_args()
    return config


config = vars(parse_args())
print(config)


test_dataset = NucleiDataset(path=config['path'], dataset=config['dataset'], split='test')
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size =1, num_workers=config['num_workers'])

models_path = ['models/pannuke_linknet.pth' ,
               'models/pannuke_unet.pth',
               'models/pannuke_unetplus.pth',
               'models/pannuke_pan.pth',
               'models/pannuke_pspnet.pth',]

models = []
for model_path in models_path:
    model = torch.load(model_path)
    model = nn.DataParallel(model.module)
    models.append(model)


def postprocess(image, h, w):
    """
        Resize uncertainty map. This is strictly for visualisation purposes.
        The output of this function will not be used for anything other
        than visualisation.
        """
    image = Image.fromarray(image)
    image = image.resize((w, h), resample=Image.BICUBIC)
    return np.array(image)

for i, (image, gt_mask) in enumerate(test_dataloader):
    image = image.cuda()
    pred_masks = []
    #models_id = []
    for m in  models:
        pred_mask = m.module.predict(image)
        pred_masks.append(pred_mask)

    image = image[0].cpu().numpy().transpose(1, 2, 0)
    #image = image[:, :,0]

    gt_mask = gt_mask[0].cpu().numpy().transpose(1, 2, 0)
    gt_mask = gt_mask[:, :, 0]



    p_mask = [pred[0].cpu().detach().numpy().transpose(1, 2, 0) for pred in pred_masks]
    predictions = [pred[:, :, 0] for pred in p_mask]

    fig, ax_arr = plt.subplots(1, 7, figsize=(20,6))
    cmap = matplotlib.cm.tab20
    cmap.set_bad(color='k')
    ax1, ax2, ax3, ax4, ax5, ax6, ax7 = ax_arr.ravel()

    ax1.imshow(image)
    ax1.set_title('Original image')
    ax1.axis('off')

    ax2.contour(np.flipud(gt_mask), colors='red')
    ax2.set_title('Mask')
    ax2.axis('off')

    ax2.imshow(image)
    ax2.contour(gt_mask, colors='red', linewidths=0.3)
    ax2.contour(predictions[0], colors='blue', linewidths=0.3)
    ax3.imshow(predictions[0], cmap=cmap)
    ax3.set_title('Unet')
    ax3.axis('off')

    ax4.imshow(predictions[1], cmap=cmap)
    ax4.set_title('Unet-plus')
    ax4.axis('off')

    ax5.imshow(predictions[2], cmap=cmap)
    ax5.set_title('PAN')
    ax5.axis('off')

    ax6.imshow(predictions[3], cmap=cmap)
    ax6.set_title('PSP Net')
    ax6.axis('off')

    ax7.imshow(predictions[4], cmap=cmap)
    ax7.set_title('Link net')
    ax7.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join('pannuke_results', '{}.jpg'.format(i)))
    plt.show()

    if i==0:
        break








