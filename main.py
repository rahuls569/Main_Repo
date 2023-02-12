import albumentations as A
import torch
import torchvision
import torchvision.transforms as transforms
import cv2 
import os
import torch.nn.functional as F
import cv2

from albumentations import Compose, PadIfNeeded, RandomCrop, Normalize, HorizontalFlip, Cutout

augmentations = Compose([
    PadIfNeeded(40),
    RandomCrop(32,32),
    HorizontalFlip(),
    Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=[0.5*255, 0.5*255, 0.5*255], always_apply=True, p=1.00),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

def load_cifar10(root, augmentations=None):
    train_transforms = torchvision.transforms.Compose([
        PadIfNeeded(40),
        RandomCrop(32,32),
        HorizontalFlip(),
        Cutout(num_holes=1, max_h_size=16, max_w_size=16, fill_value=[0.5*255, 0.5*255, 0.5*255], always_apply=True, p=1.00),
        torchvision.transforms.ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=train_transforms)

    if augmentations is not None:
        def augment(image, target):
            augmented = augmentations(image=image)
            image = augmented['image']
            return image, target

        trainset.transform = augment
    
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
