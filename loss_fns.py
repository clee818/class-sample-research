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
        
    def forward(self, inputs, targets):
        return torchvision.ops.sigmoid_focal_loss(inputs, targets, gamma=self.gamma, reduction=self.reduction)
    

class SoftmaxFocalLoss(nn.Module): 
    def __init__(self, weight=None, 
                 gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.alpha = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        log_prob = F.log_softmax(inputs, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            targets, 
            weight=self.alpha,
            reduction = self.reduction)
    
class CappedBCELoss(nn.Module):
    def __init__(self, smote_labels=None, loss_cap=1e-3, reduction='none'):
        nn.Module.__init__(self)
        self.loss_cap = loss_cap
        self.smote_labels = smote_labels
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # BCE with logits (sigmoid --> nll) --> reduction 
        loss = F.binary_cross_entropy_with_logits(input, targets)
        capped_loss = np.maximum(loss[self.smote_labels == 1], self.loss_cap, reduction=self.reduction)
       # indices = np.where(self.smote_labels == 1)[0]
        #np.where(loss[indices]
        return F.sigmoid(loss) 
                
        
