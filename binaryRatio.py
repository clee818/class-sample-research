import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class Ratio(Dataset): # start with binary classification? 
    def __init__(self, original_dataset, num_classes, target_ratios, nums=(0,1)): # target_ratios is a list   
        assert len(target_ratios) == num_classes

        images = original_dataset.train_data
        labels = original_dataset.train_labels


        class_images = []
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