import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from imblearn.over_sampling import SMOTE  
import random

class Reduce(Dataset):
    # reduces number of classes
    # takes in original dataset, target # of classes
    def __init__(self, original_dataset, num_classes, nums=(0,1), transform=None, CIFAR=False):        
       
        if CIFAR: 
            indices = np.isin(original_dataset.targets, nums) 
            self.images = torch.from_numpy(original_dataset.data[indices==1]).float()
            self.labels = torch.from_numpy(np.array(original_dataset.targets)[indices==1])
        else:
            indices = np.isin(original_dataset.train_labels, nums) 
            self.images = original_dataset.train_data[indices==1].unsqueeze(1).float()
            self.labels = original_dataset.train_labels[indices==1]
        
        self.nums = nums
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
                 
    def __getitem__(self, index): 
        image = self.images[index].float()
        if self.labels[index]==self.nums[0]:
            label = 0
        elif self.labels[index]==self.nums[1]:
            label = 1
        else: # nums[2] if it exists
            label = 2
        if self.transform:
            image = self.transform(image)
        return (image, label)
    
    


class Ratio(Dataset):
    # assume all classes are balanced 
    # takes in reduced dataset 
    def __init__(self, original_dataset, num_classes, target_ratios, nums=(3,2,1), CIFAR=True, transform=None):
        assert len(target_ratios) == num_classes
       
        self.nums=nums
        
        class_indices = np.isin(original_dataset.targets, nums)
        
        targets = np.asarray(original_dataset.targets)[class_indices]
        images = original_dataset.data[class_indices]
        
        _, class_counts = np.unique(np.sort(targets), return_counts=True)
        
        max_index = target_ratios.index(max(target_ratios))
        
        updated_ratios = tuple(ratio/target_ratios[max_index] for ratio in target_ratios)
        
        ratio_class_counts = tuple(int(ratio*class_count) for ratio, class_count in zip(updated_ratios, class_counts))
        
               
        reduced_images = []
        reduced_labels = []
                
        for i, num in enumerate(nums):
            class_images = images[(targets == num)]
            if CIFAR: 
                class_images = torch.from_numpy(class_images)
            indices = np.random.choice(class_images.shape[0], ratio_class_counts[i], replace=False)
            reduced_images.append(class_images[indices])
            reduced_labels.append(torch.from_numpy(np.full(ratio_class_counts[i], i)))
        
        self.images = torch.cat(reduced_images)
        self.labels = torch.cat(reduced_labels).int()
        self.transform=transform
     
    def __len__(self):
        return len(self.labels)
                 
    def __getitem__(self, index): 
        image = self.images[index].float()
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return (image, label) 
    

NO_SMOTE_LABEL = 0
SMOTE_LABEL = 1


class Smote(Dataset): 
    def __init__(self, ratio_dataset, target_shape, CIFAR=True, transform=None):
        
        shape = ratio_dataset.images.shape
                
        smote = SMOTE()
        
        self.images, self.labels = smote.fit_resample(ratio_dataset.images.reshape(shape[0], -1), ratio_dataset.labels)
        
        self.smote_labels = np.zeros(target_shape)
        
        self.smote_labels[shape[0]:] = SMOTE_LABEL 
        
        if CIFAR:
            self.images = torch.from_numpy(self.images.reshape(-1, shape[1], shape[2], shape[3]))
        else: 
            self.images = torch.from_numpy(self.images.reshape(-1, shape[1], shape[2]))
        
        self.labels = torch.from_numpy(self.labels)
        
        self.transform=transform
       
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index): 
        image = self.images[index].float()
        label = self.labels[index]
        smote_label = self.smote_labels[index]
        if self.transform:
            image = self.transform(image)
        return (image, label, smote_label)

    
    
class ForTripletLoss(Dataset): 
    def __init__(self, dataset, smote=False, num_classes=2, transform=None):
        self.images = dataset.images.float()
        self.labels = dataset.labels
        self.smote = smote 
       
        if smote: 
            self.smote_labels = dataset.smote_labels
            class0_smote_mask = np.full_like(self.labels, fill_value=False, dtype=bool)
            class1_smote_mask = np.full_like(self.labels, fill_value=False, dtype=bool)
            
            class0_smote_mask[self.smote_labels==NO_SMOTE_LABEL] = True
            class1_smote_mask[self.smote_labels==NO_SMOTE_LABEL] = True
            
            class0_smote_mask[self.labels!=0] = False
            class1_smote_mask[self.labels!=1] = False
            
            self.class0_images = self.images[class0_smote_mask]
            self.class1_images = self.images[class1_smote_mask]
            
            if num_classes == 3:
                class2_smote_mask = np.full_like(self.labels, fill_value=False, dtype=bool)
                class2_smote_mask[self.smote_labels==NO_SMOTE_LABEL] = True
                class2_smote_mask[self.labels!=2] = False
                self.class2_images = self.images[class2_smote_mask]
            
        else:
            self.class0_images = self.images[self.labels==0]
            self.class1_images = self.images[self.labels==1]
            
        self.transform=transform 
        self.num_classes = num_classes
        
         
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        anchor_image = self.images[index]
        anchor_label = self.labels[index]
       
        if self.num_classes==2:
            if anchor_label == 0:
                pos_image = random.choice(self.class0_images)
                neg_image = random.choice(self.class1_images)
            else:
                pos_image = random.choice(self.class1_images)
                neg_image = random.choice(self.class0_images)
        elif self.num_classes == 3:
            if anchor_label == 0:
                pos_image = random.choice(self.class0_images)
                neg_image = random.choice(torch.cat((self.class1_images, self.class2_images)))
            elif anchor_label == 1:
                pos_image = random.choice(self.class1_images)
                neg_image = random.choice(torch.cat((self.class0_images, self.class2_images)))
            else:
                pos_image = random.choice(self.class2_images)
                neg_image = random.choice(torch.cat((self.class0_images, self.class1_images)))
            
        
        if self.transform:
            anchor_image = self.transform(anchor_image)
            pos_image = self.transform(pos_image)
            neg_image = self.transform(neg_image)
        if self.smote: 
            anchor_smote_label = self.smote_labels[index]
            return (anchor_image, pos_image, neg_image, anchor_label, anchor_smote_label)
        return (anchor_image, pos_image, neg_image, anchor_label)