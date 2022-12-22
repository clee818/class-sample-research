import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
from sklearn import metrics

def test(test_loader, network, verbose=True, loss_fn=nn.NLLLoss(), return_acc=False):
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
          #  test_loss += loss_fn(output.squeeze().float(), target.float()).item()
            test_loss += loss_fn(output.squeeze(), target).item()
    
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            preds.extend(pred)
            unbinarized_preds.extend(output.data)
            true.extend(target)
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
  #  auc = metrics.roc_auc_score(true, unbinarized_preds)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    test_loss, correct, len(test_loader.dataset),
    accuracy))
   # print(f'\nTest set: Avg. loss: {test_loss}, AUC: {auc} \n')  
    if return_acc:
        return test_losses, accuracy
    return test_losses, preds, true