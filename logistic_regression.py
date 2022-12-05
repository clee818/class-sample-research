import torch 
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        
        self.fc = nn.Linear(784, num_classes)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc(x)
        return F.log_softmax(x)