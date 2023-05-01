import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import loss_fns
import numpy as np

LOG_INTERVAL = 10

def train_sigmoid(epoch, train_loader, network, optimizer, directory=None, verbose=True, loss_fn=nn.BCEWithLogitsLoss, loss_fn_args = {}, smote=False):
    train_counter = []
    train_losses = []
    
    loss_fn=loss_fn(**loss_fn_args)
    
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        if smote:
            loss = loss_fn(output.squeeze().float(), (target[0].float(), target[1].float()))
        else:
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

def train_sigmoid_with_embeddings(epoch, train_loader, network, optimizer, directory=None, verbose=True, loss_fn=loss_fns.CappedBCELoss, loss_fn_args = {}, smote=True):
    train_counter = []
    train_losses = []
    
    loss_func = loss_fn(**loss_fn_args)
    
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        
        output, embeds = network(data)
        
        if (not loss_fn_args['loss_cap']) and (batch_idx > 5):
            dist = euclidean_distance(embeds, target[0])
            loss_fn_args['loss_cap'] == 10 * torch.exp(dist / 15) # big distance = small cap 
            loss_func=loss_fn(**loss_fn_args) 
            loss_fn_args['loss_cap'] == None
            
        if smote:
            loss = loss_func(output.squeeze().float(), (target[0].float(), target[1].float()))
        else:
            loss = loss_func(output.squeeze().float(), target.float())
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


def euclidean_distance(x, y): 
    #NOTE: ONLY WORKS FOR BINARY CLASSIFICATION RIGHT NOW
    # computes euclidean distance between each example and the average 
    dist = torch.zeros(x.shape[0])
    targets = np.array(y)
    for label in range(2):
        avg = torch.mean(x[targets==label], 0) 
        dist[targets==label] = (x[targets==label] - avg).pow(2).sum(1).sqrt()
    return dist


def train_softmax(epoch, train_loader, network, optimizer, directory=None, verbose=True, loss_fn=nn.CrossEntropyLoss, loss_fn_args={}, smote=False):
    train_counter = []
    train_losses = []

    network.train()
    
    loss_fn = loss_fn(**loss_fn_args)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        if smote:
            loss = loss_fn(output.squeeze(), (target[0].long(), target[1].long()))
        else:
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


