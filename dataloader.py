import os
import torch
import glob
import numpy as np
import pandas as pd
from skimage import io, transform
from torchvision import transforms
import torchvision.transforms.functional as F 
from torch.utils.data import Dataset
from PIL import Image

class GPSDataset(Dataset):
    def __init__(self, metadata, root_dir, transform = None):
        self.metadata = pd.read_csv(metadata)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        folder_idx = self.metadata.iloc[idx, 0]    
        image_root_path = "{}{}".format(self.root_dir, folder_idx)
        images = np.stack([io.imread("{}/{}".format(image_root_path, x)) / 255.0 for x in os.listdir(image_root_path)]) 
                
        sample = {'images': images, 'folder_idx': folder_idx}        
        if self.transform:
            sample['images'] = self.transform(sample['images'])
            
        return sample

    
class GPSReducedDataset(Dataset):
    def __init__(self, metadata, root_dir, predict_y_idx):
        self.metadata = pd.read_csv(metadata)
        self.root_dir = root_dir
        self.predict_y_idx = predict_y_idx

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        folder_idx = self.metadata.iloc[idx, 0]     
        feature_matrix = np.genfromtxt("{}{}.csv".format(self.root_dir, folder_idx), delimiter=' ')       
        sample = {'images': feature_matrix, 'y': torch.Tensor(self.metadata.iloc[idx, 1:].values.astype(float))}   
        sample['y'] = sample['y'][self.predict_y_idx]
        return sample


class ProxyDataset(Dataset):
    def __init__(self, metadata, root_dir, transform = None):
        self.metadata = pd.read_csv(metadata)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_id, y_urban, y_rural, y_env = self.metadata.iloc[idx, :].values
        image_path = "{}{}.png".format(self.root_dir, int(image_id))
        image = io.imread(image_path) / 255.0
        
        sample = {'image': image, 'y': torch.Tensor([y_urban, y_rural, y_env])}     
        if self.transform:
            sample['image'] = self.transform(np.stack([image])).squeeze()
            
        return sample   

class UnlabeledDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.file_list = glob.glob('./{}/*.png'.format(root_dir))
        self.root_dir = './{}/'.format(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        images = Image.open(self.file_list[idx])
        if self.transform:
            images = self.transform(images)
        return images    
    

class RemovalDataset(Dataset):
    def __init__(self, metadata, root_dir, transform = None):
        self.metadata = pd.read_csv(metadata)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        image_id, y_env = self.metadata.iloc[idx, :].values
        image_path = "{}{}.png".format(self.root_dir, int(image_id))
        image = io.imread(image_path) / 255.0
        
        sample = {'image': image, 'y': y_env}      
        if self.transform:
            sample['image'] = self.transform(np.stack([image])).squeeze()
            
        return sample       


class RandomRotate(object):
    def __call__(self, images):
        rotated = np.stack([self.random_rotate(x) for x in images])
        return rotated
    
    def random_rotate(self, image):
        rand_num = np.random.randint(0, 4)
        if rand_num == 0:
            return np.rot90(image, k=1, axes=(0, 1))
        elif rand_num == 1:
            return np.rot90(image, k=2, axes=(0, 1))
        elif rand_num == 2:
            return np.rot90(image, k=3, axes=(0, 1))   
        else:
            return image


class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, images):
        normalized = np.stack([F.normalize(x, self.mean, self.std, self.inplace) for x in images]) 
        return normalized
        

class ToTensor(object):
    def __call__(self, images):
        images = images.transpose((0, 3, 1, 2))
        return torch.from_numpy(images).float()     
   