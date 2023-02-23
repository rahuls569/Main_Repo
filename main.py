import albumentations as A
import torch
import torchvision
import torchvision.transforms as transforms
import cv2 
import os
import torch.nn.functional as F
import cv2

import matplotlib.pyplot as plt
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm

augmentations = [A.HorizontalFlip(), A.ShiftScaleRotate(),
                 A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16,
                                 min_width=16, fill_value=(0.5, 0.5, 0.5), mask_fill_value=None),
                 A.Cutout(num_holes=1, max_h_size=16, max_w_size=16, p=1.0)]

def load_cifar10(root, augmentations=None):
  train_transforms = transforms.Compose([
                                    #  transforms.Resize((28, 28)),
                                   #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                     transforms.RandomCrop(32, padding=4),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     # Note the difference between (0.1307) and (0.1307,)
                                    ])

  test_transforms = transforms.Compose([
                                     transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])

  trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=train_transforms)

  if augmentations is not None:
      def augment(image):
          image = transforms.ToPILImage()(image)
          for aug in augmentations:
              image = aug(image=image)["image"]
          image = transforms.ToTensor()(image)
          return image
      trainset.Transform = augment
    
  testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=test_transforms)

  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  return trainset, testset, classes

import torch
import torchvision
import torchvision.transforms as transforms

def setup_dataloaders(trainset, testset, SEED, Batch):
   
    cuda = torch.cuda.is_available()
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)
    
    dataloader_args = dict(shuffle=True, batch_size=Batch, num_workers=2, pin_memory=True) if cuda else dict(shuffle=True, batch_size=Batch)
    
    train_loader = torch.utils.data.DataLoader(trainset, **dataloader_args)
    test_loader = torch.utils.data.DataLoader(testset, **dataloader_args)
    
    return train_loader, test_loader 
  
  
  
  
from tqdm import tqdm
class Trainer:
    def __init__(self):
        self.train_losses = []
        
        self.train_acc = []
        

    def train(self, model, device, train_loader, optimizer, criterion, epoch):
        model.train()
        pbar = tqdm(train_loader)
        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar):
            # get samples
            data, target = data.to(device), target.to(device)

            # Init
            optimizer.zero_grad()

            # Predict
            y_pred = model(data)

            # Calculate loss
            loss = criterion(y_pred, target)
            self.train_losses.append(loss.item())

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update pbar-tqdm
            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
            self.train_acc.append(100*correct/processed)
            
class Test:
    def __init__(self):
        self.test_losses = []
        self.test_acc = []
        self.misclassified_images = []
        self.classified_images = []

    def test(self, model, device, test_loader, criterion):
        model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                
                # compare predictions with true label
                for i, (p, t) in enumerate(zip(pred, target)):
                    if p != t:
                        self.misclassified_images.append((data[i], p, t))
                    else:
                        self.classified_images.append((data[i], p, t))
                
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        self.test_losses.append(test_loss)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        self.test_acc.append(100. * correct / len(test_loader.dataset))
        return self.misclassified_images, self.classified_images
      
      
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
def trainNetwork(net, device, train_loader, test_loader, EPOCHS, lr=0.2):
  Trainer1= Trainer()
  tester  = Test()
 
  optimizer = optim.SGD(net.parameters(), lr, momentum=0.9)
  scheduler = StepLR(optimizer, step_size=6, gamma=0.1)
  criterion = nn.CrossEntropyLoss()
 
  for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    Trainer1.train(net, device, train_loader, optimizer, criterion, epoch)
    scheduler.step()
    tester.test(net, device, test_loader, criterion)     
  return Trainer1, tester



class CIFAR10Dataset:
    def __init__(self, root_path):
        self.root_path = root_path
        self.dataset = datasets.CIFAR10(root_path, train=True, download=True)
        self.data = self.dataset.data
        self.targets = self.dataset.targets
        self.classes = self.dataset.classes

    def get_data(self):
        return self.data

    def get_targets(self):
        return self.targets

    def get_classes(self):
        return self.classes




