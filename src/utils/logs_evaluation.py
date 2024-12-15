# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 10:29:02 2021

@author: rajgudhe
"""

import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt


logs = glob.glob('evaluation_results/pannuke/*')
print(logs)

for data in logs:
    experiment_name = os.path.splitext(os.path.split(data)[-1])[0]
    df = pd.read_csv(data, header=None, delimiter='\t')


    #precision = df[].tolist()
    #recall = df[13].tolist()
    #accuracy = df[7].tolist() #14
    fscore = df[1].tolist()
    IoU = df[2].tolist()
    pq_score = df[3].tolist()

    print('__________________________________________________________________________________________________________________')
    print('Experiment:{}'.format(experiment_name))
    #print(str('Test Average precision: {} +/- {}'.format(str(np.mean(precision)*100), str(np.std(precision)))))
    #print(str('Test Average recall: {} +/- {}'.format(str(np.mean(recall)*100), str(np.std(recall)))))
    print(str('Test Average fscore:  {} +/- {}'.format(str(np.mean(fscore) * 100), str(np.std(fscore)))))
    print(str('Test Average IoU:  {} +/- {}'.format(str(np.mean(IoU) * 100), str(np.std(IoU)))))
    print(str('Test Average PQ:  {} +/- {}'.format(str(np.mean(pq_score) * 100), str(np.std(pq_score)))))
    print('__________________________________________________________________________________________________________________')


