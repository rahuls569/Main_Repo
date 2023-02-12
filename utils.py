'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init


class ComputeMeanAndStd:
    def __init__(self, trainset, testset):
        self.trainset = trainset
        self.testset = testset

    def get_mean_and_std(self, dataset_type):
        '''Compute the mean and std value of dataset.'''
        if dataset_type == 'train':
            dataset = self.trainset
        elif dataset_type == 'test':
            dataset = self.testset
        else:
            raise ValueError("Invalid dataset type. Choose 'train' or 'test'")
            
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
        mean = torch.zeros(3)
        std = torch.zeros(3)
        print(f'==> Computing mean and std for {dataset_type}set..')
        for inputs, targets in dataloader:
            for i in range(3):
                mean[i] += inputs[:,i,:,:].mean()
                std[i] += inputs[:,i,:,:].std()
        mean.div_(len(dataset))
        std.div_(len(dataset))
        return mean, std


