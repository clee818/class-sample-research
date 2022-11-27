from torch.utils.data import Dataset, DataLoader

class reduce(Dataset):
    # reduces number of classes
    # takes in original dataset, target # of classes
    def __init__(self, original_dataset, num_classes):        
        
        indices = original_dataset.train_labels < num_classes
        self.images = original_dataset.train_data[indices==1].unsqueeze(1)
        self.labels = original_dataset.train_labels[indices==1]
        
                 
    def __len__(self):
        return len(self.labels)
                 
    def __getitem__(self, index): 
        image = self.images[index].float()
        return (image, self.labels[index])
