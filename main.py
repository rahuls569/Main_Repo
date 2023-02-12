import albumentations as A
import torch
import torchvision
import torchvision.transforms as transforms
import cv2 
import os
import torch.nn.functional as F
import cv2

from albumentations import Compose, PadIfNeeded, RandomCrop, Normalize, HorizontalFlip, ShiftScaleRotate, CoarseDropout, Cutout, PadIfNeeded
from albumentations.pytorch.transforms import ToTensorV2

augmentations = [A.HorizontalFlip(), A.ShiftScaleRotate(),
                 A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16,
                                 min_width=16, fill_value=(0.5, 0.5, 0.5), mask_fill_value=None)]

def load_cifar10(root, augmentations=None):
  train_transforms = transforms.Compose([
                                    #  transforms.Resize((28, 28)),
                                   #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                      PadIfNeeded(40),
                                      RandomCrop(32,32),
                                      HorizontalFlip(),
                                      Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=[0.4914*255, 0.4822*255, 0.4471*255], always_apply=True, p=1.00),
                                     transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
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
