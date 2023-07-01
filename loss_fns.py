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
    def __init__(self, loss_cap=None, distance=None, reduction='mean', cap_array=None):
        nn.Module.__init__(self)
        self.loss_cap = loss_cap
        self.distance = distance
        self.reduction = reduction
        self.cap_array = cap_array # used if cap already calculated 
        
    def cosine_distance(self, data, targets, smote_targets):
        dist = torch.zeros(data.shape[0])
        targets = np.array(targets)
        for label in range(2):
            mask = np.zeros(targets.shape)
            mask[targets == label] = 1
            mask[smote_targets == SMOTE_LABEL] = 0
            avg = torch.mean(data[mask == 1], 0)
            dist[targets == label] = 1 - F.cosine_similarity(data[targets == label], avg)
        return dist

    def euclidean_distance(self, data, targets, smote_targets):
        dist = torch.zeros(data.shape[0])
        targets = np.array(targets)
        for label in range(2):
            mask = np.zeros(targets.shape)
            mask[targets == label] = 1
            mask[smote_targets == SMOTE_LABEL] = 0
            avg = torch.mean(data[mask == 1], 0)
            dist[targets == label] = (data[targets == label] - avg).pow(2).sum(
                1).sqrt()
        return dist
        
        
    def forward(self, inputs, targets, smote_targets, embeds=None):
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        if self.loss_cap != None:
            if self.cap_array != None:
                # capped smote using triplet loss
                loss[smote_targets == SMOTE_LABEL] = torch.minimum(loss[smote_targets == SMOTE_LABEL], torch.tensor(self.cap_array * self.loss_cap)[smote_targets==SMOTE_LABEL])
            elif self.distance == 'euclidean':
                # euclidean distance 
                distances = self.euclidean_distance(embeds, targets, smote_targets)
                cap = self.loss_cap / distances 
                loss[smote_targets == SMOTE_LABEL] = torch.minimum(loss[smote_targets == SMOTE_LABEL], torch.tensor(cap)[smote_targets==SMOTE_LABEL])
            elif self.distance == 'cosine':
                # cosine distance 
                distances = self.cosine_distance(embeds, targets, smote_targets) 
                cap = self.loss_cap / distances
                loss[smote_targets == SMOTE_LABEL] = torch.minimum(loss[smote_targets == SMOTE_LABEL], torch.tensor(cap)[smote_targets==SMOTE_LABEL])
            else:
                # cap is a constant 
                loss[smote_targets == SMOTE_LABEL] = torch.minimum(loss[smote_targets == SMOTE_LABEL], torch.tensor(self.loss_cap))
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
        losses[losses == 0] = 0.00001
        return losses