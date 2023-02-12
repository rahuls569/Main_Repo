'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init

import numpy as np

class ImageData:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def data(self, train_flag):
        if train_flag:
            return self.train_data
        else:
            return self.test_data

    def data_summary_stats(self):
        # Load train data as numpy array
        train_data = self.data(train_flag=True)
        test_data = self.data(train_flag=False)

        total_data = np.concatenate((train_data, test_data), axis=0)
        print(total_data.shape)
        print(total_data.mean(axis=(0,1))/255)
        print(total_data.std(axis=(0,1))/255)
