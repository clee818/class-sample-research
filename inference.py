import torch
import torch.nn as nn


def run_inference_sigmoid(dataloader, network, binary=True):  
    losses = []
    y_preds = []
    unbinarized_preds = []
    y_true = []
    
    network.eval()
    loss = 0
    correct = 0
        
    
    with torch.no_grad():
        for data, target in dataloader:
            output = network(data)
            if binary:
                loss_fn=nn.BCELoss()
                loss += loss_fn(output.squeeze().float(), target.float()).item()
                pred=output.data
            else:
                loss_fn=nn.CrossEntropyLoss()
                loss += loss_fn(output.squeeze(), target).item()
                pred=output.data.max(1, keepdim=True)[1]
            y_preds.extend(pred.float())
            y_true.extend(target.float())
        loss /= len(dataloader.dataset)
        losses.append(loss)
    return losses, y_preds, y_true

def run_inference_softmax(dataloader, network, loss_fn=nn.CrossEntropyLoss()): 
    losses = []
    
    y_preds = []
    unbinarized_preds = []
    y_true = []
    
    network.eval()
    loss = 0
    correct = 0
    
    
    with torch.no_grad():
        for data, target in dataloader:
            output = network(data)
            loss += loss_fn(output.squeeze(), target).item()
            pred = output.data.max(1, keepdim=True)[1]
            y_preds.extend(pred)
            y_true.extend(target)
        loss /= len(dataloader.dataset)
        losses.append(loss)
    return losses, y_preds, y_true
