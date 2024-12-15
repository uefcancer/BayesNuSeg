# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 08:09:58 2020

@author: rajgudhe
"""

import argparse
import os
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import NucleiDatasetOrgan
import segmentation_models as smp

# AdrenalGland  Bile-duct Bladder Breast Cervix Colon Esophagus
# HeadNeck Kidney Liver Lung Overian Paccreatic Prostate Skin Stomach Testis Thyroid Uterus
# LymphNodes  Mediastinum   Pancreas   Pleura   Skin   Testes   Thymus   ThyroidGland

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='datasets', type=str, help='dataset root path')
    parser.add_argument('--dataset', default='PanNuke', type=str, help='Mammogram view')
    parser.add_argument('--organ', default='Lung', type=str, help='name of the organ')

    parser.add_argument('-tb', '--test_batch_size', default=1, type=int, metavar='N',
                        help='mini-batch size (default: 16)')
    parser.add_argument('--num_epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of workers')
    parser.add_argument('--loss_fucntion', default='JaccardLoss', type=str, help='loss fucntion')
    parser.add_argument('--model_save_path', default='models/pannuke/pannuke_pspnet.pth', type=str,
                        help='path to save the model')  # change here

    parser.add_argument('--results_path', default='evaluation_results/pannuke', type=str,
                        help='path to save the model')  # change here

    config = parser.parse_args()

    return config


config = vars(parse_args())
print(config)

test_dataset = NucleiDatasetOrgan(path=config['path'], dataset=config['dataset'], organ=config['organ'])
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size =1, num_workers=config['num_workers'])

# load best saved checkpoint
model = torch.load(config['model_save_path'])
model = nn.DataParallel(model.module)

loss = getattr(smp.utils.losses, config['loss_fucntion'])()

metrics = [
    smp.utils.metrics.Fscore(),
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.metrics.PanopticQuality(threshold=0.5)
]

DEVICE = 'cuda'
# evaluate model on test set
test_epoch = smp.utils.train.ValidEpoch(
    model=model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
)

with open(os.path.join(config['results_path'], config['organ'] + '.txt'), 'a+') as logs_file:
    for i in range(0, config['num_epochs']):
        print('\nEpoch: {}'.format(i))

        test_logs = test_epoch.run(test_dataloader)
        #print(test_logs)

        print('{} \t {} \t {} \t {} '.format(i,  test_logs['fscore'],
                                             test_logs['iou_score'],
                                             test_logs['pq_score'],
                                             ), file=logs_file)

