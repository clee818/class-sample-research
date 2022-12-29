import torch 
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
    

class SigmoidLogisticRegression(nn.Module):
    def __init__(self, num_classes):
        super(SigmoidLogisticRegression, self).__init__()

        self.num_classes = num_classes 
        if num_classes == 2: 
            self.fc = nn.Linear(784, 1)
        else: 
            self.fc = nn.Linear(784, num_classes)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc(x)
        return self.sigmoid(x)



class SoftmaxLogisticRegression(nn.Module):
    def __init__(self, num_classes):
        super(SoftmaxLogisticRegression, self).__init__()
        self.fc = nn.Linear(784, num_classes)        

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc(x)
        return F.log_softmax(x)
