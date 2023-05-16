import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import loss_fns
import numpy as np
import loss_fns

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


def train_sigmoid_with_smote(epoch, train_loader, network, optimizer, directory=None, verbose=True, loss_fn=loss_fns.CappedBCELoss, loss_fn_args = {}):
    train_counter = []
    train_losses = []
    
    loss_fn=loss_fn(**loss_fn_args)
    
    network.train()
    for batch_idx, (data, target, smote_label) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = loss_fn(output.squeeze().float(), target, smote_label)
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

def train_sigmoid_with_embeddings(epoch, train_loader, network, optimizer, directory=None, verbose=True, loss_fn=loss_fns.CappedBCELoss, loss_fn_args = {}):

    # always uses smote 

    train_counter = []
    train_losses = []
    
    loss_func = loss_fn(**loss_fn_args)
    
    network.train()
    for batch_idx, (data, target, smote_target) in enumerate(train_loader):
        optimizer.zero_grad()
        
        output, embeds = network(data)
        
        if (not loss_fn_args['loss_cap']) and (batch_idx > 5):
            dist = euclidean_distance(embeds, target)
            loss_fn_args['loss_cap'] == 10 / torch.exp(dist / 15) # big distance = small cap 
            loss_func=loss_fn(**loss_fn_args) 
            loss_fn_args['loss_cap'] == None

        loss = loss_func(output.squeeze().float(), target, smote_target)
        pred = output.data
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0 and verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
        train_counter.append((batch_idx*64) + ((epoch)*len(train_loader.dataset)))
    if directory: 
        torch.save(network.state_dict(), directory)
    return train_counter, train_losses


def euclidean_distance(data, targets): 
    # NOTE: ONLY WORKS FOR BINARY CLASSIFICATION RIGHT NOW
    # computes euclidean distance between each example and the average for that class 
    dist = torch.zeros(data.shape[0])
    targets = np.array(targets)
    for label in range(2):
        avg = torch.mean(data[targets==label], 0) 
        dist[targets==label] = (data[targets==label] - avg).pow(2).sum(1).sqrt()
    return dist


def train_triplet_loss(epoch, train_loader, network, optimizer, directory=None, verbose=True, loss_fn_args = {}):
    train_counter = []
    train_losses = []
    
    loss_fn=loss_fns.TripletLoss(**loss_fn_args)
    
    network.train()
    for batch_idx, (anchor_data, pos_data, neg_data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        _, anchor_embeds = network(anchor_data.float())
        _, pos_embeds = network(pos_data.float()) 
        _, neg_embeds = network(neg_data.float())
        loss = loss_fn(anchor_embeds, pos_embeds, neg_embeds)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0 and verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(anchor_data),len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
        train_counter.append((batch_idx*64) + ((epoch)*len(train_loader.dataset)))
    if directory: 
        torch.save(network.state_dict(), directory)
    return train_counter, train_losses

def train_linear_probe(epoch, train_loader, embed_network, linear_probe_network, optimizer, directory=None, verbose=True, loss_fn=nn.BCEWithLogitsLoss, loss_fn_args = {}):
    train_counter = []
    train_losses = []
    
    loss_fn=loss_fn(**loss_fn_args)
    
    network.train() 
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        with torch.no_grad(): 
            _, embeds = embed_network(data)
        output = linear_probe_network(embeds)
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

def train_softmax_with_smote(epoch, train_loader, network, optimizer, directory=None, verbose=True, loss_fn=loss_fns.CappedCELoss, loss_fn_args={}):
    train_counter = []
    train_losses = []

    network.train()
    
    loss_fn = loss_fn(**loss_fn_args)
    
    for batch_idx, (data, target, smote_target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = loss_fn(output.squeeze(), target, smote_target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0 and verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
        train_losses.append(loss.item())
        train_counter.append((batch_idx*64) + ((epoch)*len(train_loader.dataset)))
    if directory: 
        torch.save(network.state_dict(), directory)
    return train_counter, train_losses



