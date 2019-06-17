import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import glob
import multiprocessing as mp
import os

def loader(path, batch_size=32, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    '''
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.RandomSizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)
    '''
    indices = list(range(50000))
    np.random.shuffle(indices)
    indices = indices[:1000]


    return data.DataLoader(
        datasets.CIFAR10(path, train=True, download=True,
                             transform = transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.RandomSizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        sampler = SubsetRandomSampler(indices),
        shuffle=False,
        num_workers=mp.cpu_count(),
        pin_memory=pin_memory)

def test_loader(path, batch_size=32, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    '''
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)
    '''

    indices = list(range(10000))
    np.random.shuffle(indices)
    indices = indices[:100]

    return data.DataLoader(
        datasets.CIFAR10(path, train=False, download=True,
                             transform = transforms.Compose([
                                 transforms.Scale(256),
                                 transforms.RandomSizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=False,
        sampler = SubsetRandomSampler(indices),
        num_workers=mp.cpu_count(),
        pin_memory=pin_memory)