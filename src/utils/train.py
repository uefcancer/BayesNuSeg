# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:34:32 2020

@author: rajgudhe
"""

import argparse
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import NucleiDataset
import segmentation_models as smp


# from torchsummary import summary

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='datasets', type=str, help='dataset root path')
    parser.add_argument('--dataset', default='NuCLS', type=str, help='Mammogram view')

    parser.add_argument('-tb', '--train_batch_size', default=12, type=int, metavar='N',
                        help='mini-batch size (default: 16)')
    parser.add_argument('-vb', '--valid_batch_size', default=12, type=int, metavar='N',
                        help='mini-batch size (default: 4)')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of workers')

    parser.add_argument('--segmentation_model', default='Unet', type=str, help='Segmentation model Unet/FPN')
    parser.add_argument('--encoder', default='resnet101', type=str,
                        help='encoder name resnet18, vgg16.......')  # change here
    parser.add_argument('--pretrained_weights', default='imagenet', type=str, help='imagenet weights')
    parser.add_argument('--activation_function', default='sigmoid', type=str,
                        help='activation of the final segmentation layer')

    parser.add_argument('--loss_fucntion', default='JaccardLoss', type=str, help='loss fucntion')
    parser.add_argument('--optimizer', default='Adam', type=str, help='optimization')
    # parser.add_argument('--lr_schedular', default='Adam', type=str, help='lr_schedular')
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of epochs')

    parser.add_argument('--logs_file_path', default='logs/nucls_unet.txt', type=str,
                        help='path to save logs')  # change here
    parser.add_argument('--model_save_path', default='models/nucls_unet.pth', type=str,
                        help='path to save the model')  # change here

    config = parser.parse_args()

    return config


config = vars(parse_args())
print(config)

train_dataset = NucleiDataset(path=config['path'], dataset=config['dataset'], split='train')
valid_dataset = NucleiDataset(path=config['path'], dataset=config['dataset'], split='val')

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config['train_batch_size'],
                              num_workers=config['num_workers'])
valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=config['valid_batch_size'],
                              num_workers=config['num_workers'])

DEVICE = 'cuda'

# create segmentation model with pretrained encoder

model = getattr(smp, config['segmentation_model'])(
    encoder_name=config['encoder'],
    encoder_weights=config['pretrained_weights'],
    classes=1,
    activation=config['activation_function'])

model = model.cuda()
model = nn.DataParallel(model)
# print(summary(model, (3, 256, 256)))


loss = getattr(smp.utils.losses, config['loss_fucntion'])()

metrics = [
    smp.utils.metrics.Accuracy(),
    smp.utils.metrics.Fscore(),
    smp.utils.metrics.IoU(threshold=0.5),
    smp.utils.metrics.PanopticQuality(threshold=0.5)

]

optimizer = getattr(torch.optim, config['optimizer'])([
    dict(params=model.parameters(), lr=config['lr']),
])

lr_schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2)
# lr_schedular = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#lr_schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0, last_epoch=-1)

# create epoch runners
# it is a simple loop of iterating over dataloader`s samples
train_epoch = smp.utils.train.TrainEpoch(
    model,
    loss=loss,
    metrics=metrics,
    optimizer=optimizer,
    lr_schedular=lr_schedular,
    device=DEVICE,
    verbose=True,
)

valid_epoch = smp.utils.train.ValidEpoch(
    model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    verbose=True,
)

with open(config['logs_file_path'], 'a+') as logs_file:
    # train model for 40 epochs

    max_score = 0
    for i in range(0, config['num_epochs']):
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_dataloader)
        valid_logs = valid_epoch.run(valid_dataloader)

        print('{} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} '.format(i, train_logs['jaccard_loss'],
                                                                           train_logs['fscore'],
                                                                           train_logs['accuracy'],
                                                                           train_logs['iou_score'],
                                                                           train_logs['pq_score'],
                                                                           valid_logs['jaccard_loss'],
                                                                           valid_logs['fscore'],
                                                                           valid_logs['accuracy'],
                                                                           valid_logs['iou_score'],
                                                                           valid_logs['pq_score'],
                                                                           ), file=logs_file)

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, config['model_save_path'])
            print('Model saved!')




