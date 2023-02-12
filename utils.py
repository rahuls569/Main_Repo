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
  



def Graph_loss_accuracy(Trainer1, tester, EPOCHS):

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
    
    
    
def plot_misclassified_images(misclassified_images, classes):
    fig, axs = plt.subplots(5, 2, figsize=(10,10))
    axs = axs.ravel()

    for i, (img, pred, true) in enumerate(misclassified_images[:10]):
        img = img.permute(1, 2, 0)
        axs[i].imshow(img.cpu().squeeze(), cmap='gray')
        axs[i].set_title(f'Actual: {classes[true]}, Predicted: {classes[pred]}')
        axs[i].axis('off')
    plt.subplots_adjust(hspace=0.5)
    plt.show()

    
 def Class_Accuracy(net, dataloader, classes, device):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images,labels = images.to(device),labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    
    
    
'''GradCAM in PyTorch.
Grad-CAM implementation in Pytorch
Reference:
[1] https://github.com/vickyliin/gradcam_plus_plus-pytorch
[2] The paper authors torch implementation: https://github.com/ramprs/grad-cam
'''

layer_finders = {}


def register_layer_finder(model_type):
    def register(func):
        layer_finders[model_type] = func
        return func
    return register


def visualize_cam(mask, img, alpha=1.0):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) * alpha

    result = heatmap+img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result


@register_layer_finder('resnet')
def find_resnet_layer(arch, target_layer_name):
    """Find resnet layer to calculate GradCAM and GradCAM++
    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'conv1'
            target_layer_name = 'layer1'
            target_layer_name = 'layer1_basicblock0'
            target_layer_name = 'layer1_basicblock0_relu'
            target_layer_name = 'layer1_bottleneck0'
            target_layer_name = 'layer1_bottleneck0_conv1'
            target_layer_name = 'layer1_bottleneck0_downsample'
            target_layer_name = 'layer1_bottleneck0_downsample_0'
            target_layer_name = 'avgpool'
            target_layer_name = 'fc'
    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    if 'layer' in target_layer_name:
        hierarchy = target_layer_name.split('_')
        layer_num = int(hierarchy[0].lstrip('layer'))
        if layer_num == 1:
            target_layer = arch.layer1
        elif layer_num == 2:
            target_layer = arch.layer2
        elif layer_num == 3:
            target_layer = arch.layer3
        elif layer_num == 4:
            target_layer = arch.layer4
        else:
            raise ValueError('unknown layer : {}'.format(target_layer_name))

        if len(hierarchy) >= 2:
            bottleneck_num = int(hierarchy[1].lower().lstrip('bottleneck').lstrip('basicblock'))
            target_layer = target_layer[bottleneck_num]

        if len(hierarchy) >= 3:
            target_layer = target_layer._modules[hierarchy[2]]

        if len(hierarchy) == 4:
            target_layer = target_layer._modules[hierarchy[3]]

    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer


def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)


def normalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)


class GradCAM:
    """Calculate GradCAM salinecy map.
    Args:
        input: input image with shape of (1, 3, H, W)
        class_idx (int): class index for calculating GradCAM.
                If not specified, the class index that makes the highest model prediction score will be used.
    Return:
        mask: saliency map of the same spatial dimension with input
        logit: model output
    A simple example:
        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        gradcam = GradCAM.from_config(model_type='resnet', arch=resnet, layer_name='layer4')
        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)
        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)
        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)
    """

    def __init__(self, arch: torch.nn.Module, target_layer: torch.nn.Module):
        self.model_arch = arch

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]

        def forward_hook(module, input, output):
            self.activations['value'] = output

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    @classmethod
    def from_config(cls, arch: torch.nn.Module, model_type: str, layer_name: str):
        target_layer = layer_finders[model_type](arch, layer_name)
        return cls(arch, target_layer)

    def saliency_map_size(self, *input_size):
        device = next(self.model_arch.parameters()).device
        self.model_arch(torch.zeros(1, 3, *input_size, device=device))
        return self.activations['value'].shape[2:]

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        # alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)

def Grad_CAM(net, testloader, classes, device):


    net.eval()

    misclassified_images = []
    actual_labels = []
    predicted_labels = []
    
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            _, pred = torch.max(output, 1)
            for i in range(len(pred)):
                if pred[i] != target[i]:
                    misclassified_images.append(data[i])
                    actual_labels.append(target[i])
                    predicted_labels.append(pred[i])

    gradcam = GradCAM.from_config(model_type='resnet', arch=net, layer_name='layer4')

    fig = plt.figure(figsize=(5, 20))
    idx_cnt=1
    for idx in np.arange(10):

        img = misclassified_images[idx]
        lbl = predicted_labels[idx]
        lblp = actual_labels[idx]

     
        img = img.unsqueeze(0).to(device)
        org_img = denormalize(img,mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5))

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(img, class_idx=lbl)
       
        heatmap, cam_result = visualize_cam(mask, org_img, alpha=0.4)

        # Show images
        # for idx in np.arange(len(labels.numpy())):
        # Original picture
        
        ax = fig.add_subplot(10, 3, idx_cnt, xticks=[], yticks=[])
        npimg = np.transpose(org_img[0].cpu().numpy(),(1,2,0))
        ax.imshow(npimg, cmap='gray')
        ax.set_title(f"Label={str(classes[lblp])}\npred={classes[lbl]}")
        idx_cnt+=1

        ax = fig.add_subplot(10, 3, idx_cnt, xticks=[], yticks=[])
        npimg = np.transpose(heatmap,(1,2,0))
        ax.imshow(npimg, cmap='gray')
        ax.set_title("HeatMap".format(str(classes[lbl])))
        idx_cnt+=1

        ax = fig.add_subplot(10, 3, idx_cnt, xticks=[], yticks=[])
        npimg = np.transpose(cam_result,(1,2,0))
        ax.imshow(npimg, cmap='gray')
        ax.set_title("GradCAM".format(str(classes[lbl])))
        idx_cnt+=1

    fig.tight_layout()  
    plt.show()
    
    
    
    
def plot_classified_images(classified_images, classes):
    fig, axs = plt.subplots(5, 2, figsize=(10,10))
    axs = axs.ravel()

    for i, (img, pred, true) in enumerate(classified_images[:10]):
        img = img.permute(1, 2, 0)
        axs[i].imshow(img.cpu().squeeze(), cmap='gray')
        axs[i].set_title(f'Actual: {classes[true]}, Predicted: {classes[pred]}')
        axs[i].axis('off')
    plt.subplots_adjust(hspace=0.5)
    plt.show()
