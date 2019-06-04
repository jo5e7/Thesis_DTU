import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision.models import densenet169

from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()

        # get the pretrained DenseNet201 network
        self.densenet = densenet169(pretrained=True)

        # disect the network to access its last convolutional layer
        self.features_conv = self.densenet.features

        # get the classifier of the densenet
        self.classifier = self.densenet.classifier

        # placeholder for the gradients
        self.gradients = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, no_grad=False):
        x = self.features_conv(x)
        # register the hook
        if (no_grad is False) and (self.training is False):
            h = x.register_hook(self.activations_hook)
        x = F.relu(x)




        # don't forget the pooling
        #print(x)
        #print(features.size(0))
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features_conv(x)