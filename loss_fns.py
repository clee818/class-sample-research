import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.ops


# from https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8


class SigmoidFocalLoss(nn.Module):
    def __init__(self, weight=None, 
                 gamma=2., reduction='none'):
        nn.Module.__init__(self)
       # self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, x, targets): # x and targets are tensors 
        return torchvision.ops.sigmoid_focal_loss(x, targets, gamma=self.gamma, reduction=self.reduction)
   
