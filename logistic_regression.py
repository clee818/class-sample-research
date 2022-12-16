import torch 
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        
        self.num_classes = num_classes 
        if num_classes == 2: 
            self.fc = nn.Linear(784, 1)
        else: 
            self.fc = nn.Linear(784, num_classes)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc(x)
        if self.num_classes == 2:
            return self.sigmoid(x)
        return F.log_softmax(x)