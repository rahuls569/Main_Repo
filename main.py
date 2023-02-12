import albumentations as A
import torch
import torchvision
import torchvision.transforms as transforms
import cv2 

augmentations = [A.HorizontalFlip(), A.ShiftScaleRotate(),
                 A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16,
                                 min_width=16, fill_value=(0.5, 0.5, 0.5), mask_fill_value=None)]

def load_cifar10(root, augmentations=None):
  train_transforms = transforms.Compose([
                                    #  transforms.Resize((28, 28)),
                                   #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                 
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
