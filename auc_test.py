import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
from sklearn import metrics

def sigmoid_test(test_loader, network, verbose=True, loss_fn=nn.NLLLoss()):
    test_losses = []
    
    preds = []
    unbinarized_preds = []
    true = []
    
    network.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += loss_fn(output.squeeze().float(), target.float()).item()
            pred = output.data
            correct += pred.eq(target.data.view_as(pred)).sum()
            preds.extend(pred.float())
            true.extend(target.float())
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
    auc = metrics.roc_auc_score(true, preds)
    print(f'\nTest set: Avg. loss: {test_loss}, AUC: {auc} \n')    
    return test_losses, auc



def softmax_test(test_loader, network, verbose=True, loss_fn=nn.CrossEntropyLoss(), datatype="float"):
    test_losses = []
    
    preds = []
    unbinarized_preds = []
    true = []
    
    network.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += loss_fn(output.squeeze(), target).item()
            pred = (torch.exp(output.data))[:, 1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            preds.extend(pred.float())
            true.extend(target.float())
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
    auc = metrics.roc_auc_score(true, preds)
    print(f'\nTest set: Avg. loss: {test_loss}, AUC: {auc} \n')    
    return test_losses, auc