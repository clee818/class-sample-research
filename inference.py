import torch
import torch.nn as nn


def run_inference_sigmoid(dataloader, network):  
    losses = []
    y_preds = []
    unbinarized_preds = []
    y_true = []
    
    network.eval()
    loss = 0
    correct = 0
    
    loss_fn=nn.BCELoss()

        
    
    with torch.no_grad():
        for data, target in dataloader:
            output = network(data)
            loss += loss_fn(output.squeeze().float(), target.float()).item()
            pred=output.data
            y_preds.extend(pred.float())
            y_true.extend(target.float())
        loss /= len(dataloader.dataset)
        losses.append(loss)
    return losses, y_preds, y_true

def run_inference_softmax(dataloader, network): 
    losses = []
    
    y_preds = []
    unbinarized_preds = []
    y_true = []
    
    network.eval()
    loss = 0
    correct = 0
    
    loss_fn=nn.CrossEntropyLoss()
    
    
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
