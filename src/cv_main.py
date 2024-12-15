import os
import argparse
import numpy as np
import torch
import pandas as pd
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import NucleiDataset
import segmentation_models as smp
from sklearn.model_selection import KFold



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='data', type=str, help='dataset root path')
    parser.add_argument('--dataset', default='pannuke', type=str, help='dataset name')

    parser.add_argument('-tb', '--train_batch_size', default=12, type=int, metavar='N',
                        help='mini-batch size (default: 16)')
    parser.add_argument('-vb', '--valid_batch_size', default=12, type=int, metavar='N',
                        help='mini-batch size (default: 4)')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers')

    parser.add_argument('--segmentation_model', default='DeepLabV3Plus', type=str, help='Segmentation model Unet/UnetPlusPlus/Linknet/FPN/PSPNet/PAN/DeepLabV3/DeepLabV3Plus')
    parser.add_argument('--encoder', default='resnet18', type=str,
                        help='encoder name resnet18, vgg16.......')  # change here
    parser.add_argument('--pretrained_weights', default='imagenet', type=str, help='imagenet weights')
    parser.add_argument('--activation_function', default='sigmoid', type=str,
                        help='activation of the final segmentation layer')

    parser.add_argument('--loss_fucntion', default='FocalTverskyLoss', type=str, choices=['JaccardLoss', 'DiceLoss','TverskyLoss', 'FocalTverskyLoss'], help='loss fucntion')
    parser.add_argument('--optimizer', default='Adam', type=str, help='optimization')
    # parser.add_argument('--lr_schedular', default='Adam', type=str, help='lr_schedular')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--num_epochs', default=10, type=int, help='Number of epochs')

    parser.add_argument('--logs_file_path', default='logs/pannuke_unet.txt', type=str,
                        help='path to save logs')  # change here
    parser.add_argument('--model_save_path', default='models/pannuke_unet.pth', type=str,
                        help='path to save the model')  # change here

    config = parser.parse_args()

    return config


config = vars(parse_args())
print(config)

# Determine the device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


kf = KFold(n_splits=5, shuffle=True, random_state=42)

dataset = NucleiDataset(path=config['path'], dataset=config['dataset'])

# Declare lists to hold metrics for all folds
iou_scores = []
accuracies = []
fscores = []
pq_scores = []
ua = []

# Initialize log dataframe
log_df = pd.DataFrame(columns=['Fold', 'Epoch', 'Accuracy', 'F1-score', 'IoU', 'PQ', 'UA'])


for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print('Fold: {}'.format(fold))
    train_fold = torch.utils.data.Subset(dataset, train_idx)
    val_fold = torch.utils.data.Subset(dataset, val_idx)
    
    train_dataloader = DataLoader(train_fold, shuffle=True, batch_size=config['train_batch_size'], num_workers=config['num_workers'])
    valid_dataloader = DataLoader(val_fold, batch_size=config['valid_batch_size'], num_workers=config['num_workers'])
    
    model = getattr(smp, config['segmentation_model'])(
        encoder_name=config['encoder'],
        encoder_weights=config['pretrained_weights'],
        classes=1,
        activation=config['activation_function'])
    
    model = model.to(device)
    model = nn.DataParallel(model)

    loss = getattr(smp.utils.losses, config['loss_fucntion'])()
    loss = loss.to(device)

    metrics = [
        smp.utils.metrics.Accuracy(),
        smp.utils.metrics.Fscore(),
        smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.PanopticQuality(threshold=0.5),
        smp.utils.metrics.UA(threshold=0.5)]

    optimizer = getattr(torch.optim, config['optimizer'])([dict(params=model.parameters(), lr=config['lr']),])
    lr_schedular = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.25, patience=2)

    train_epoch = smp.utils.train.TrainEpoch(model,loss=loss, metrics=metrics, optimizer=optimizer,lr_schedular=lr_schedular, device=device,verbose=True)
    valid_epoch = smp.utils.train.ValidEpoch(model,loss=loss,metrics=metrics,device=device,verbose=True,)

    max_score = 0
    for i in range(0, config['num_epochs']):
        print('\n Epoch: {}'.format(i))
        train_logs = train_epoch.run(train_dataloader)
        valid_logs = valid_epoch.run(valid_dataloader)

        new_row = pd.DataFrame({'Fold': [fold],
                         'Epoch': [i],
                         'Accuracy': [valid_logs['accuracy']],
                         'F1-score': [valid_logs['fscore']],
                         'IoU': [valid_logs['iou_score']],
                         'PQ': [valid_logs['pq_score']],
                         'UA':[valid_logs['ua']]})

        log_df = pd.concat([log_df, new_row], ignore_index=True)


        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, f"{config['model_save_path']}_fold_{fold}")
            print('Model saved!')

    # Append metrics of this fold to the lists
    iou_scores.append(valid_logs['iou_score'])
    accuracies.append(valid_logs['accuracy'])
    fscores.append(valid_logs['fscore'])
    pq_scores.append(valid_logs['pq_score'])
    ua.append(valid_logs['ua'])

log_df.to_csv('logs/deeplab3_plus.csv', index=False, header=True)

# Compute mean and standard deviation for each metric
mean_iou = np.mean(iou_scores)
std_iou = np.std(iou_scores)

mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

mean_fscore = np.mean(fscores)
std_fscore = np.std(fscores)

mean_pq = np.mean(pq_scores)
std_pq = np.std(pq_scores)

mean_ua = np.mean(ua)
std_ua = np.std(ua)

print(f"Mean Accuracy: {mean_accuracy}, Std Dev: {std_accuracy}")
print(f"Mean F-score: {mean_fscore}, Std Dev: {std_fscore}")
print(f"Mean IoU: {mean_iou}, Std Dev: {std_iou}")
print(f"Mean PQ: {mean_pq}, Std Dev: {std_pq}")
print(f"Mean UA: {mean_ua}, Std Dev: {std_ua}")
