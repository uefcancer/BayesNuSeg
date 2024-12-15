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
from matplotlib import ticker


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='datasets', type=str, help='dataset root path')
    parser.add_argument('--dataset', default='PanNuke', type=str, help='Mammogram view')

    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers')
    parser.add_argument('--model_save_path', default='models/pannuke/pannuke_pspnet.pth', type=str,
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
#print(model)

def entropy(p, eps=1e-6):
    p = torch.clamp(p, eps, 1.0-eps)
    return -1.0*((p*torch.log(p)) + ((1.0-p)*(torch.log(1.0-p))))

def expected_entropy(mc_preds):
    return torch.mean(entropy(mc_preds), dim=0)

def predictive_entropy(mc_preds):
    return entropy(torch.mean(mc_preds, dim=0))


def monte_carlo_sampling(model, image, n_samples):
    mc_predcitions = []

    for i in range(n_samples):
        pred_mask = model.module.predict(image)
        mc_predcitions.append(pred_mask)

    return torch.cat(mc_predcitions, dim=0)
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
    print(i, image.shape)
    image = image.cuda()
    pred_mask = model.module.predict(image)
    pred_mask = pred_mask[0].cpu().detach().numpy().transpose(1, 2, 0)
    pred_mask = pred_mask[:, :, 0]


    output = monte_carlo_sampling(model, image, n_samples=50)

    result = uncertaininty_map(image, gt_mask, model, 50)
    print(result.shape)
    map = postprocess_uncertainty(result, 256, 256)


    mean_out = torch.mean(output, dim=0)
    mean_out = mean_out.cpu().numpy().transpose(1, 2, 0)
    mean_out = mean_out[:, :, 0]
    print(mean_out.shape)

    image = image[0].cpu().numpy().transpose(1, 2, 0)

    gt_mask = gt_mask[0].cpu().numpy().transpose(1, 2, 0)
    gt_mask = gt_mask[:, :, 0]

    print(gt_mask.shape)


    # uncertaininty
    total = predictive_entropy(output)
    aleatoric = expected_entropy(output)
    epistemic = total -aleatoric

    aleatoric = aleatoric[0].cpu().detach().numpy()
    epistemic = epistemic[0].cpu().detach().numpy()
    total = total[0].cpu().detach().numpy()

    total1 = postprocess_uncertainty(total, 256,256)
    epistemic1 = postprocess_uncertainty(epistemic, 256,256)
    aleatoric1 = postprocess_uncertainty(aleatoric, 256,256)

    error = (gt_mask != pred_mask).astype(np.uint8)
    #print(error.shape)


    fig, ax_arr = plt.subplots(1,4, figsize=(16,5))
    #fig.suptitle('Uncertaininy', fontsize=16)
    ax1, ax2, ax3, ax4  = ax_arr.ravel()

    ax1.imshow(image)
    ax1.contour(gt_mask, colors='red', linewidths=0.4)
    ax1.contour(mean_out, colors='blue', linewidths=0.4)
    #ax1.set_title('Image \n', fontsize=16, fontweight='bold')
    ax1.axis('off')

    ax2.matshow(-epistemic1, cmap='jet')
    #ax2.set_title('Epistemic', fontsize=16, fontweight='bold')
    ax2.axis('off')

    ax3.matshow(aleatoric1, cmap='jet')
    #ax3.set_title('Aleatoric',  fontsize=16, fontweight='bold')
    ax3.axis('off')

    divider1 = make_axes_locatable(ax4)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    im = ax4.matshow(map, cmap='jet')
    #ax4.set_title('Total Uncertainty ', fontsize=16, fontweight='bold')
    ax4.axis('off')
    cbar = plt.colorbar(im, cax=cax1, orientation='vertical')







    """
    #ax3.contourf(np.flipud(total),  cmap='jet')

    divider1 = make_axes_locatable(ax4)
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    im1 = ax4.matshow(aleatoric1, cmap='jet')
    ax4.set_title('Aleatoric Uncertaininty',  fontsize=16)
    ax4.axis('off')
    cbar1 = plt.colorbar(im1, cax=cax1, orientation='vertical')
    #cbar.ax.text(1, 0, 'Blue: Less uncertaininty', fontsize=14)

    divider2 = make_axes_locatable(ax5)
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    im2 = ax5.matshow(total1, cmap='jet')
    ax5.set_title('Uncertaininty', fontsize=16)
    ax5.axis('off')
    cbar2 = plt.colorbar(im2, cax=cax2, orientation='vertical')

    """
    plt.tight_layout()

    plt.savefig(os.path.join('pannuke_uncertainity', '{}.jpg'.format(i)))
    #plt.show()





