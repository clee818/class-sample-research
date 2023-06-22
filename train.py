import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import loss_fns
import numpy as np
import loss_fns

LOG_INTERVAL = 10


def train_sigmoid(epoch, train_loader, network, optimizer, directory=None,
                  verbose=True, loss_fn=nn.BCEWithLogitsLoss, loss_fn_args={}):
    train_counter = []
    train_losses = []

    loss_fn = loss_fn(**loss_fn_args)

    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = loss_fn(output.squeeze().float(), target.float())
        pred = output.data
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0 and verbose:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                         batch_idx * len(data),
                                                                         len(train_loader.dataset),
                                                                         100. * batch_idx /
                                                                         len(train_loader),
                                                                         loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx * 64) + ((epoch) * len(train_loader.dataset)))
    if directory:
        torch.save(network.state_dict(), directory)
    return train_counter, train_losses


def train_sigmoid_with_smote(epoch, train_loader, network, optimizer,
                             directory=None, verbose=True,
                             loss_fn=loss_fns.CappedBCELoss, loss_fn_args={}):
    train_counter = []
    train_losses = []

    loss_fn = loss_fn(**loss_fn_args)

    network.train()
    for batch_idx, (data, target, smote_label) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = loss_fn(output.squeeze().float(), target.float(), smote_label)
        pred = output.data
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0 and verbose:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                         batch_idx * len(
                                                                             data),
                                                                         len(train_loader.dataset),
                                                                         100. * batch_idx /
                                                                         len(train_loader),
                                                                         loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx * 64) + ((epoch) * len(train_loader.dataset)))
    if directory:
        torch.save(network.state_dict(), directory)
    return train_counter, train_losses


def train_sigmoid_with_embeddings(epoch, train_loader, network, optimizer,
                                  directory=None, verbose=True,
                                  loss_fn=loss_fns.CappedBCELoss,
                                  loss_fn_args={}):
    # always uses smote

    train_counter = []
    train_losses = []

    loss_func = loss_fn(**loss_fn_args)

    network.train()
    for batch_idx, (data, target, smote_target) in enumerate(train_loader):
        optimizer.zero_grad()

        output, embeds = network(data)
        
        loss = loss_func(output.squeeze().float(), target.float(), smote_target)
        pred = output.data
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0 and verbose:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                         batch_idx * len(
                                                                             data),
                                                                         len(train_loader.dataset),
                                                                         100. * batch_idx /
                                                                         len(train_loader),
                                                                         loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx * 64) + ((epoch) * len(train_loader.dataset)))
    if directory:
        torch.save(network.state_dict(), directory)
    return train_counter, train_losses

def train_sigmoid_euclidean_distance(epoch, train_loader, network, optimizer,
                                  directory=None, verbose=True,
                                  loss_fn=loss_fns.CappedBCELoss,
                                  loss_fn_args={}):
    # always uses smote

    train_counter = []
    train_losses = []

    loss_func = loss_fn(**loss_fn_args)

    network.train()
    for batch_idx, (data, target, smote_target) in enumerate(train_loader):
        optimizer.zero_grad()

        output, embeds = network(data)

        if (batch_idx > 5):
            dist = euclidean_distance(embeds, target, smote_target)
            loss_fn_args['loss_cap'] = 10 / torch.exp(
                dist / 15)  # big distance = small cap
            loss_func = loss_fn(**loss_fn_args)
            loss_fn_args['loss_cap'] = None 
        loss = loss_func(output.squeeze().float(), target.float(), smote_target)
        pred = output.data
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0 and verbose:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                         batch_idx * len(
                                                                             data),
                                                                         len(train_loader.dataset),
                                                                         100. * batch_idx /
                                                                         len(train_loader),
                                                                         loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx * 64) + ((epoch) * len(train_loader.dataset)))
    if directory:
        torch.save(network.state_dict(), directory)
    return train_counter, train_losses

def euclidean_distance(data, targets, smote_targets):
    # NOTE: ONLY WORKS FOR BINARY CLASSIFICATION RIGHT NOW
    # computes euclidean distance between each example and the average for that class 
    # excludes SMOTE targets 
    dist = torch.zeros(data.shape[0])
    targets = np.array(targets)
    for label in range(2):
        mask = np.zeros(targets.shape)
        mask[targets == label] = 1
        mask[smote_targets == 1] = 0
        avg = torch.mean(data[mask == 1], 0)
        dist[targets == label] = (data[targets == label] - avg).pow(2).sum(
            1).sqrt()
    return dist


def train_sigmoid_cosine_distance(epoch, train_loader, network, optimizer,
                                  directory=None, verbose=True,
                                  loss_fn=loss_fns.CappedBCELoss,
                                  loss_fn_args={}):
    # always uses smote

    train_counter = []
    train_losses = []

    loss_func = loss_fn(**loss_fn_args)

    network.train()
    for batch_idx, (data, target, smote_target) in enumerate(train_loader):
        optimizer.zero_grad()

        output, embeds = network(data)

        if (batch_idx > 5):
            dist = cosine_distance(embeds, target, smote_target)
            loss_fn_args['loss_cap'] = (1 / dist) * 5 
            loss_func = loss_fn(**loss_fn_args)
            loss_fn_args['loss_cap'] = None 
        loss = loss_func(output.squeeze().float(), target.float(), smote_target)
        pred = output.data
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0 and verbose:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                         batch_idx * len(
                                                                             data),
                                                                         len(train_loader.dataset),
                                                                         100. * batch_idx /
                                                                         len(train_loader),
                                                                         loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx * 64) + ((epoch) * len(train_loader.dataset)))
    if directory:
        torch.save(network.state_dict(), directory)
    return train_counter, train_losses




def cosine_distance(data, targets, smote_targets):
    dist = torch.zeros(data.shape[0])
    targets = np.array(targets)
    for label in range(2):
        mask = np.zeros(targets.shape)
        mask[targets == label] = 1
        mask[smote_targets == 1] = 0
        avg = torch.mean(data[mask == 1], 0)
        F.cosine_similarity
        dist[targets == label] = 1 - F.cosine_similarity(data[targets == label], avg)
    return dist
    

def train_triplet_loss(epoch, train_loader, network, optimizer, directory=None,
                       verbose=True, loss_fn_args={}):
    train_counter = []
    train_losses = []

    loss_fn = loss_fns.TripletLoss(**loss_fn_args)
   

    network.train()
    for batch_idx, (anchor_data, pos_data, neg_data, target) in enumerate(
            train_loader):
        optimizer.zero_grad()
        anchor_embeds = network(anchor_data.float())
        pos_embeds = network(pos_data.float())
        neg_embeds = network(neg_data.float())
        loss = loss_fn(anchor_embeds, pos_embeds, neg_embeds)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0 and verbose:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                         batch_idx * len(
                                                                             anchor_data),
                                                                         len(train_loader.dataset),
                                                                         100. * batch_idx / len(
                                                                             train_loader),
                                                                         loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx * 64) + ((epoch) * len(train_loader.dataset)))
    if directory:
        torch.save(network.state_dict(), directory)
    return train_counter, train_losses


def train_linear_probe(epoch, train_loader, network,
                       optimizer, directory=None, verbose=True,
                       loss_fn=nn.BCEWithLogitsLoss, loss_fn_args={}):
    
    train_counter = []
    train_losses = []

    loss_fn = loss_fn(**loss_fn_args)

    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  
        output, _ = network(data)
        loss = loss_fn(output.squeeze().float(), target.float())
        pred = output.data
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0 and verbose:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                         batch_idx * len(
                                                                             data),
                                                                         len(train_loader.dataset),
                                                                         100. * batch_idx /
                                                                         len(train_loader),
                                                                         loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx * 64) + ((epoch) * len(train_loader.dataset)))
    if directory:
        torch.save(network.state_dict(), directory)
    return train_counter, train_losses

import matplotlib.pyplot as plt

def train_triplet_capped_loss(epoch, train_loader, network, optimizer, directory=None,
                       verbose=True, loss_fn=loss_fns.CappedBCELoss, loss_fn_args={}):
    train_counter = []
    train_losses = []

    cap_calc = loss_fns.TripletLoss(reduction='none')
    loss_func = loss_fn(**loss_fn_args)

    same_count = 0
    diff_count = 0
    
    network.train()
    for batch_idx, (anchor_data, pos_data, neg_data, target, smote_target) in enumerate(
            train_loader):
        optimizer.zero_grad()
        anchor_output, anchor_embeds = network(anchor_data.float())
        if (batch_idx > 5):
            _, pos_embeds = network(pos_data.float())
            _, neg_embeds = network(neg_data.float())
            cap, distance_positive, distance_negative = cap_calc(anchor_embeds, pos_embeds, neg_embeds)

            for i in range(len(cap)):
                if distance_positive[i] == distance_negative[i]:
                    same_count+=1 
                else:
                    diff_count += 1

            loss_fn_args['loss_cap'] = 1 / cap
            loss_func = loss_fn(**loss_fn_args)
            loss_fn_args['loss_cap'] = None 
        loss = loss_func(anchor_output.squeeze(), target.float(), smote_target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0 and verbose:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                         batch_idx * len(
                                                                             anchor_data),
                                                                         len(train_loader.dataset),
                                                                         100. * batch_idx / len(
                                                                             train_loader),
                                                                         loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx * 64) + ((epoch) * len(train_loader.dataset)))
    if directory:
        torch.save(network.state_dict(), directory)
    
    print(same_count / (same_count + diff_count))
    return train_counter, train_losses


def train_softmax(epoch, train_loader, network, optimizer, directory=None,
                  verbose=True, loss_fn=nn.CrossEntropyLoss, loss_fn_args={}):
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
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                         batch_idx * len(
                                                                             data),
                                                                         len(train_loader.dataset),
                                                                         100. * batch_idx /
                                                                         len(train_loader),
                                                                         loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx * 64) + ((epoch) * len(train_loader.dataset)))
    if directory:
        torch.save(network.state_dict(), directory)
    return train_counter, train_losses


def train_softmax_with_smote(epoch, train_loader, network, optimizer,
                             directory=None, verbose=True,
                             loss_fn=loss_fns.CappedCELoss, loss_fn_args={}):
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
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                         batch_idx * len(
                                                                             data),
                                                                         len(train_loader.dataset),
                                                                         100. * batch_idx /
                                                                         len(train_loader),
                                                                         loss.item()))
        train_losses.append(loss.item())
        train_counter.append(
            (batch_idx * 64) + ((epoch) * len(train_loader.dataset)))
    if directory:
        torch.save(network.state_dict(), directory)
    return train_counter, train_losses
