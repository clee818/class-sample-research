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
    
class SoftmaxFocalLoss(nn.Module): 
     
    def __init__(self, weight=None, 
                 gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.alpha = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target_tensor, 
            weight=self.alpha,
            reduction = self.reduction
        )
    """
    def __init__(self, gamma=2., weights=1):
        nn.Module.__init__(self)
        self.gamma = gamma
        self.alpha = weights
     
    def forward(self, x, target): 
        n = x.shape[0]
        range_n = torch.arange(0, n, dtype=torch.int64)
        pos_num =  float(x.shape[1])
        p = torch.softmax(x, dim=1)
        p = p[range_n, target]
        loss = -(1-p)**self.gamma*self.alpha*torch.log(p)
        return torch.sum(loss) / pos_num
        """

    
