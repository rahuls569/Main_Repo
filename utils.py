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
        
        
import matplotlib.pyplot as plt
import numpy as np
import torchvision



class YourClass:
    def __init__(self, train_loader, classes, channel_means, channel_stdevs):
        self.train_loader = train_loader
        self.classes = classes
        self.channel_means = channel_means
        self.channel_stdevs = channel_stdevs
        self.channel_stdevs = torch.tensor(self.channel_stdevs)
        self.channel_means = torch.tensor(self.channel_means)

    def unnormalize(self, img):
        img = img * self.channel_stdevs.reshape(3, 1, 1) + self.channel_means.reshape(3, 1, 1)
        return img

    def imshow(self, img):
        npimg = self.unnormalize(img).numpy()
        plt.figure(figsize=(20,20))  # increase size of plot
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def display_images(self):
        dataiter = iter(self.train_loader)
        images, labels = next(dataiter)
        images = images[:32]  # only display first 20 images
        labels = labels[:32]  # only display labels for first 20 images

        # display images
        self.imshow(torchvision.utils.make_grid(images))

        # print labels
        for i in range(32):
            index = labels[i].item()
            print('Image %d: %s' % (i, self.classes[index]))
  



def graph_loss_accuracy(Trainer1, tester, EPOCHS):

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # train_epoch_linspace = np.linspace(1, EPOCHS, len(Trainer1.train_losses))
    # test_epoch_linspace = np.linspace(1, EPOCHS, len(tester.test_losses))

    # Training Plot
    axs[0,0].plot(Trainer1.train_losses, color='r', label='Training Loss')
    axs[0,0].set_xlabel('Epochs')
    axs[0,0].set_ylabel('Training Loss')
    axs[0,0].set_title('Training Loss vs. Epochs')
    axs[0,0].legend()

    # TEST Loss Plot
    axs[0,1].plot(tester.test_losses, color='r', label='Test Loss')
    axs[0,1].set_xlabel('Epochs')
    axs[0,1].set_ylabel('Test Loss')
    axs[0,1].set_title('Test Loss vs. Epochs')
    axs[0,1].legend()

    # TRAIN Accuracy Plot
    axs[1,0].plot( Trainer1.train_acc, color='r', label='Training Accuracy')
    axs[1,0].set_xlabel('Epochs')
    axs[1,0].set_ylabel('Training Accuracy')
    axs[1,0].set_title('Training Accuracy vs. Epochs')
    axs[1,0].legend()

    # TEST Accuracy Plot
    axs[1,1].plot( tester.test_acc, color='r', label='Test Accuracy')
    axs[1,1].set_xlabel('Epochs')
    axs[1,1].set_ylabel('Test Accuracy')
    axs[1,1].set_title('Test Accuracy vs. Epochs')
    axs[1,1].legend()

    plt.show()           

