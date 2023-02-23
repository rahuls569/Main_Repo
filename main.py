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
      
      
      
      
      
      
import albumentations as A

class AlbumentationImageDataset(Dataset):
    def __init__(self, image_list, train=True):
        self.image_list = image_list
        self.train = train

        if self.train:
            self.transform = A.Compose([
                A.PadIfNeeded(min_height=36, min_width=36),
                A.RandomCrop(height=32, width=32),
                A.HorizontalFlip(p=0.5),
                A.Cutout(num_holes=1, max_h_size=8, max_w_size=8, fill_value=[0.49139968 * 255, 0.48215841 * 255, 0.44653091 * 255]),
                A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
            ])
        else:
            self.transform = A.Compose([
                A.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))
            ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        image, label = self.image_list[i]
        image = np.array(image)

        if self.train:
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            transformed = self.transform(image=image)
            image = transformed['image']

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return torch.tensor(image), label


def train(model, device, train_loader, criterion, scheduler, optimizer, use_l1=False, lambda_l1=0.01):
    """Function to train the model

    Args:
        model (instance): torch model instance of defined model
        device (str): "cpu" or "cuda" device to be used
        train_loader (instance): Torch Dataloader instance for trainingset
        criterion (instance): criterion to used for calculating the loss
        scheduler (function): scheduler to be used
        optimizer (function): optimizer to be used
        use_l1 (bool, optional): L1 Regularization method set True to use . Defaults to False.
        lambda_l1 (float, optional): Regularization parameter of L1. Defaults to 0.01.

    Returns:
        float: accuracy and loss values
    """
    model.train()
    pbar = tqdm(train_loader)
    lr_trend = []
    correct = 0
    processed = 0
    train_loss = 0

    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch 
        # accumulates the gradients on subsequent backward passes. Because of this, when you start your training loop, 
        # ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)
        # Calculate loss
        loss = criterion(y_pred, target)

        l1=0
        if use_l1:
            for p in model.parameters():
                l1 = l1 + p.abs().sum()
        loss = loss + lambda_l1*l1

        # Backpropagation
        loss.backward()
        optimizer.step()
        # updating LR
        if scheduler:
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
                lr_trend.append(scheduler.get_last_lr()[0])

        train_loss += loss.item()

        # Update pbar-tqdm
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        

        pbar.set_description(desc= f'Batch_id={batch_idx} Loss={train_loss/(batch_idx + 1):.5f} Accuracy={100*correct/processed:0.2f}%')
    return 100*correct/processed, train_loss/(batch_idx + 1), lr_trend


  
  
  
def test(model, device, test_loader, criterion):
    """put model in eval mode and test it

    Args:
        model (instance): torch model instance of defined model
        device (str): "cpu" or "cuda" device to be used
        test_loader (instance): Torch Dataloader instance for testset
        criterion (instance): criterion to used for calculating the loss

    Returns:
        float: accuracy and loss values
    """
    model.eval()
    test_loss = 0
    correct = 0
    #iteration = len(test_loader.dataset)// test_loader.batch_size
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset), test_loss


def save_model(model, epoch, optimizer, path):
    """Save torch model in .pt format

    Args:
        model (instace): torch instance of model to be saved
        epoch (int): epoch num
        optimizer (instance): torch optimizer
        path (str): model saving path
    """
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, path)


def fit_model(net, optimizer, criterion, device, NUM_EPOCHS,train_loader, test_loader, use_l1=False, scheduler=None, save_best=False):
    """Fit the model

    Args:
        net (instance): torch model instance of defined model
        optimizer (function): optimizer to be used
        criterion (instance): criterion to used for calculating the loss
        device (str): "cpu" or "cuda" device to be used
        NUM_EPOCHS (int): number of epochs for model to be trained
        train_loader (instance): Torch Dataloader instance for trainingset
        test_loader (instance): Torch Dataloader instance for testset
        use_l1 (bool, optional): L1 Regularization method set True to use. Defaults to False.
        scheduler (function, optional): scheduler to be used. Defaults to None.
        save_best (bool, optional): If save best model to model.pt file, paramater validation loss will be monitered

    Returns:
        (model, list): trained model and training logs
    """
    training_acc, training_loss, testing_acc, testing_loss = list(), list(), list(), list()
    lr_trend = []
    if save_best:
        min_val_loss = np.inf
        save_path = 'model.pt'

    for epoch in range(1,NUM_EPOCHS+1):
        print("EPOCH: {} (LR: {})".format(epoch, optimizer.param_groups[0]['lr']))
        
        train_acc, train_loss, lr_hist = train(
            model=net, 
            device=device, 
            train_loader=train_loader, 
            criterion=criterion ,
            optimizer=optimizer, 
            use_l1=use_l1, 
            scheduler=scheduler
        )
        test_acc, test_loss = test(net, device, test_loader, criterion)
        # update LR
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
        
        if save_best:
            if test_loss < min_val_loss:
                print(f'Valid loss reduced from {min_val_loss:.5f} to {test_loss:.6f}. checkpoint created at...{save_path}\n')
                save_model(net, epoch, optimizer, save_path)
                min_val_loss = test_loss
            else:
                print(f'Valid loss did not inprove from {min_val_loss:.5f}\n')
        else:
            print()

        training_acc.append(train_acc)
        training_loss.append(train_loss)
        testing_acc.append(test_acc)
        testing_loss.append(test_loss)
        lr_trend.extend(lr_hist)    

    if scheduler:   
        return net, (training_acc, training_loss, testing_acc, testing_loss, lr_trend)
    else:
        return net, (training_acc, training_loss, testing_acc, testing_loss)



