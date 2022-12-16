from torch.utils.data import Dataset, DataLoader
import numpy as np

class Reduce(Dataset):
    # reduces number of classes
    # takes in original dataset, target # of classes
    def __init__(self, original_dataset, num_classes, nums=(0,1)):        
        
        indices = np.isin(original_dataset.train_labels, nums) 
            
        self.images = original_dataset.train_data[indices==1].unsqueeze(1)
        self.labels = original_dataset.train_labels[indices==1]
        self.nums = nums
        
    def __len__(self):
        return len(self.labels)
                 
    def __getitem__(self, index): 
        image = self.images[index].float()
        if self.labels[index]==self.nums[0]:
            label = 0
        else:
            label = 1
        return (image, label)