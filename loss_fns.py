import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.ops


# from https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8

SMOTE_LABEL = 1

class SigmoidFocalLoss(nn.Module):
    def __init__(self, weight=None, 
                 gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        return torchvision.ops.sigmoid_focal_loss(inputs * self.weight, targets, gamma=self.gamma, reduction=self.reduction)
    

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
    def __init__(self, loss_cap=None, reduction='mean'):
        nn.Module.__init__(self)
        self.loss_cap = loss_cap
        self.reduction = reduction
        
        
    def forward(self, inputs, targets, smote_targets):
        # BCE with logits (sigmoid --> nll) --> reduction 
        # caps smote examples 
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        if self.loss_cap != None:
            loss[smote_targets == SMOTE_LABEL] = torch.minimum(loss[smote_targets == SMOTE_LABEL], torch.tensor(self.loss_cap)[smote_targets==SMOTE_LABEL])
              #  loss[smote_targets == SMOTE_LABEL] = torch.minimum(loss[smote_targets == SMOTE_LABEL], torch.tensor(self.loss_cap))
        if self.reduction=='mean':
            return torch.mean(loss)
        return loss
    
class AllCappedBCELoss(nn.Module):
    def __init__(self, loss_cap=None, reduction='mean'):
        nn.Module.__init__(self)
        self.loss_cap = loss_cap
        self.reduction = reduction
        
        
    def forward(self, inputs, targets, smote_targets):
        # same as CappedBCELoss but caps all examples 
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        if self.loss_cap:
            loss = torch.minimum(loss, torch.tensor(self.loss_cap))
        if self.reduction=='mean':
            return torch.mean(loss)
        return loss
    

    
class CappedCELoss(nn.Module):
    def __init__(self, loss_cap=None, reduction='mean'):
        nn.Module.__init__(self)
        self.loss_cap = loss_cap
        self.reduction = reduction
        
        
    def forward(self, inputs, targets):
        loss = F.cross_entropy(inputs, targets[0], reduction='none')
        if self.loss_cap:
            loss[targets[1] == SMOTE_LABEL] = torch.minimum(loss[targets[1] == SMOTE_LABEL], torch.tensor(self.loss_cap))
        if self.reduction=='mean':
            return torch.mean(loss)
        return loss
    

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3, reduction='mean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        
    def euclidean_distance(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.euclidean_distance(anchor, positive)
        distance_negative = self.euclidean_distance(anchor, negative)
        if self.reduction == 'mean':
            distance_positive = distance_positive.mean()
            distance_negative = distance_negative.mean()
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses