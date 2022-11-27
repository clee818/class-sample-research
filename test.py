import torch
import torch.nn as nn
import torch.nn.functional as F

def test(test_loader, network, verbose=True):
    test_losses = []
    
    preds = []
    true = []
    
    network.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            preds.extend(pred)
            true.extend(target)
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

    return test_losses, preds, true