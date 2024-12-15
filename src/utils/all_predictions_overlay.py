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
    parser.add_argument('--dataset', default='CryoNuSeg', type=str, help='Mammogram view')

    parser.add_argument('--num_workers', default=5, type=int, help='Number of workers')

    config = parser.parse_args()
    return config


config = vars(parse_args())
print(config)


test_dataset = NucleiDataset(path=config['path'], dataset=config['dataset'], split='test')
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size =1, num_workers=config['num_workers'])

models_path = ['models/cryonuseg/cryonuseg_pspnet.pth' ,
               'models/cryonuseg/cryonuseg_pan.pth',
                'models/cryonuseg/cryonuseg_fpn.pth',
               'models/cryonuseg/cryonuseg_unet.pth',
                'models/cryonuseg/cryonuseg_unetplus.pth']

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
    print(i)
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

    diff = (predictions[4] != gt_mask).astype(np.uint8)






    fig, ax_arr = plt.subplots(1, 6, figsize=(22,5))
    cmap = matplotlib.cm.tab20
    cmap.set_bad(color='k')
    ax1, ax2, ax3, ax4, ax5, ax6  = ax_arr.ravel()

    ax1.imshow(image)
    #ax1.set_title('Image', fontsize=20, fontweight= 'bold')
    ax1.axis('off')

    ax2.imshow(image, alpha=0.4)
    ax2.contour(gt_mask, colors='red', linewidths=0.3)
    ax2.contour(predictions[0], colors='blue', linewidths=0.3)
    #ax2.set_title('FCN-8', fontsize=20, fontweight= 'bold')
    ax2.axis('off')

    ax3.imshow(image, alpha=0.4)
    ax3.contour(gt_mask, colors='red', linewidths=0.3)
    ax3.contour(predictions[1], colors='blue', linewidths=0.3)
    #ax3.set_title('U-Net ', fontsize=20, fontweight= 'bold')
    ax3.axis('off')

    ax4.imshow(image, alpha=0.4)
    ax4.contour(gt_mask, colors='red', linewidths=0.3)
    ax4.contour(predictions[2], colors='blue', linewidths=0.3)
    #ax4.set_title('SegNet', fontsize=20, fontweight= 'bold')
    ax4.axis('off')

    ax5.imshow(image, alpha=0.4)
    ax5.contour(gt_mask, colors='red', linewidths=0.3)
    ax5.contour(predictions[3], colors='blue', linewidths=0.3)
    #ax5.set_title('Hovernet', fontsize=20, fontweight= 'bold')
    ax5.axis('off')

    ax6.imshow(image, alpha=0.4)
    ax6.contour(gt_mask, colors='red', linewidths=0.3)
    ax6.contour(predictions[4], colors='blue', linewidths=0.3)
    #ax6.set_title('BayesNuSeg (Ours)', fontsize=20, fontweight= 'bold')
    ax6.axis('off')


    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join('cryonuseg_results_overlay', '{}.jpg'.format(i)))












