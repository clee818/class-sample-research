import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

LOG_INTERVAL = 10

def train_sigmoid(epoch, train_loader, network, optimizer, directory=None, verbose=True, loss_fn=nn.BCEWithLogitsLoss, loss_fn_args = {}):
    train_counter = []
    train_losses = []
    
    loss_fn=loss_fn(**loss_fn_args)
    
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = loss_fn(output.squeeze().float(), target.float())
        pred=output.data
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0 and verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
        train_counter.append((batch_idx*64) + ((epoch)*len(train_loader.dataset)))
    if directory: 
        torch.save(network.state_dict(), directory)
    return train_counter, train_losses


def train_softmax(epoch, train_loader, network, optimizer, directory=None, verbose=True, loss_fn=nn.CrossEntropyLoss, loss_fn_args={}):
    train_counter = []
    train_losses = []

    network.train()
    
    loss_fn = loss_fn(**loss_fn_args)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = loss_fn(output.squeeze(), target.long())
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0 and verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
        train_counter.append((batch_idx*64) + ((epoch)*len(train_loader.dataset)))
    if directory: 
        torch.save(network.state_dict(), directory)
    return train_counter, train_losses


