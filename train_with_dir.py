import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

LOG_INTERVAL = 10


def train(epoch, train_loader, network, optimizer, directory, verbose=True):
    train_counter = []
    train_losses = []

    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
        train_counter.append((batch_idx*64) + ((epoch)*len(train_loader.dataset)))
    torch.save(network.state_dict(), directory)    
    return train_counter, train_losses