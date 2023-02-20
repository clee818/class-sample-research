import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class Reduce(Dataset):
    # reduces number of classes
    # takes in original dataset, target # of classes
    def __init__(self, original_dataset, num_classes, nums=(0,1), CIFAR=False):        
       
        if CIFAR: 
            indices = np.isin(original_dataset.targets, nums) 
            self.images = torch.from_numpy(original_dataset.data[indices==1])
            self.labels = torch.from_numpy(np.array(original_dataset.targets)[indices==1])
        else:
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
    
    
    
class BinaryRatio(Dataset): 
    def __init__(self, original_dataset, num_classes, target_ratios, nums=(0,1), CIFAR=False): # target_ratios is a list   
        assert len(target_ratios) == num_classes
        
        images = None 
        labels = None 
        
        class_images = []

        if CIFAR:
            images = torch.from_numpy(original_dataset.data)
            labels = torch.from_numpy(np.array(original_dataset.targets))
            class_images.append(images[(labels == nums[0])]) 
            class_images.append(images[(labels == nums[1])])
        else: 
            images = original_dataset.train_data
            labels = original_dataset.train_labels
            class_images.append(images[(labels == nums[0])].unsqueeze(1)) # same shape as labels w/ both classes
            class_images.append(images[(labels == nums[1])].unsqueeze(1))



        


        class_labels = []
        class_labels.append(labels[(labels == nums[0])]) 
        class_labels.append(labels[(labels == nums[1])])


        total = class_images[0].shape[0] + class_images[1].shape[0]

        # ratio, take class w/ limiting factor
        # if target_ratio > current_ratio, use class in numerator 

        class_ratios = []
        class_ratios.append(class_labels[0].shape[0]/total)
        class_ratios.append(class_labels[1].shape[0]/total)



        if (target_ratios[0]/target_ratios[1]) > (class_ratios[0]/class_ratios[1]):
            reduced_images, reduced_labels = resample(class_images[0], class_images[1], class_labels[0], class_labels[1], target_ratios[0], target_ratios[1])
            
        else:
            reduced_images, reduced_labels = resample(class_images[1], class_images[0], class_labels[1], class_labels[0], target_ratios[1], target_ratios[0])

        
        self.labels = reduced_labels
        self.images = reduced_images
        self.nums = nums 
        
        
                 
    def __len__(self):
        return len(self.labels)
                 
    def __getitem__(self, index): 
        image = self.images[index].float()
        label = self.labels[index]
        if self.labels[index]==self.nums[0]:
            label = 0
        else:
            label = 1
        return (image, label)
    
    
def resample(images1, images2, labels1, labels2, target_ratios1, target_ratios2):

    reduced_images = images1
    reduced_labels = labels1

    n = int(target_ratios2/target_ratios1 * labels1.shape[0])
    indices = np.random.choice(labels2.shape[0], n, replace=False)


    reduced_images = torch.cat((reduced_images, images2[indices]))
    reduced_labels = torch.cat((reduced_labels, labels2[indices]))
    

    return reduced_images, reduced_labels




class Ratio(Dataset):
    # try 3 classes
    # assume all classes are relatively balanced 
    def __init__(self, original_dataset, num_classes, target_ratios, nums=(3,1,2), CIFAR=True):
        assert len(target_ratios) == num_classes
       
        self.nums=nums
        
        indices = np.isin(original_dataset.targets, nums) 
        
        targets = np.array(np.asarray(original_dataset.targets)[indices])
        _, class_counts = np.unique(targets, return_counts=True)
        
        max_index = target_ratios.index(max(target_ratios))
        
        updated_ratios = tuple(ratio/target_ratios[max_index] for ratio in target_ratios)
      
        
        ratio_class_counts = tuple(int(ratio*class_count) for ratio, class_count in zip(updated_ratios, class_counts))
        
        
        
        reduced_images = torch.empty(1) 
        reduced_labels = torch.empty(1)
        
        
       # indices = np.random.choice(labels2.shape[0], n, replace=False)
        
        for i, num in enumerate(nums):
            class_images = torch.from_numpy(targets[targets==num])
            indices = np.random.choice(class_images.shape[0], ratio_class_counts[i], replace=False)
            reduced_images = torch.cat((reduced_images, class_images[indices]))
            reduced_labels = torch.cat((reduced_labels, torch.from_numpy(np.full(ratio_class_counts[i], num))))
        
        self.images = reduced_images
        self.labels = reduced_labels.int()
                                       
        
        
    def __len__(self):
        return len(self.labels)
                 
    def __getitem__(self, index): 
        image = self.images[index].float()
        label = self.nums.index(self.labels[index])
        return (image, label) 
        
        