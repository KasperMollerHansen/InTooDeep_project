# %%
# Load packages
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import nn

#%%
# Change CSV file
def change_csv_file(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    # Create 2 new columns
    df["camera_01"] = df.index.to_series().apply(lambda x: f"image_01_{x + 1}.jpg")
    df["camera_02"] = df.index.to_series().apply(lambda x: f"image_02_{x + 1}.jpg")
    # Save the new CSV file
    df.to_csv('data/rotations_w_images.csv', index=False)
change_csv_file("data/rotations.csv")

#%%
# Dataloader
class WindTurbineDataset(Dataset):
    """
        Args:
            root_dir (string): Directory with csv and image folder.
            csv_file (string): Path to the csv file with filenames and angles
            image_folder (string): Directory with images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
    def __init__(self, csv_file, image_folder, root_dir, images_num=1, transform=None):
        self.root_dir = root_dir
        csv_path =  os.path.join(self.root_dir, csv_file)
        self.rotations_df = pd.read_csv(csv_path)
        self.transform = transform
        self.image_folder = []
        for i in range(images_num):
            self.image_folder.append(image_folder+f"_0{i+1}/")

    def __len__(self):
        return len(self.rotations_df)

    def __getitem__(self, idx):
        # Get the image file path
        images = []
        for i in range(len(self.image_folder)):
            img_name = (os.path.join(self.root_dir, self.image_folder[i], self.rotations_df.iloc[idx, i+2])) #'filename' is the third column
            images.append(Image.open(img_name))
        


        base_angles = self.rotations_df.iloc[idx, 0] 
        blade_angles = self.rotations_df.iloc[idx, 1]
        angles = torch.tensor([base_angles, blade_angles], dtype=torch.float32)

        if self.transform:
            images = [self.transform(image) for image in images]
            images = torch.concatenate(images, dim=0)        
        return images, angles
    
transform = transforms.Compose([
    transforms.Resize((int(720/2),int(1280/2))), # Resizing the image to 1280x720
    transforms.ToTensor()
])


dataset = WindTurbineDataset(csv_file='rotations_w_images.csv', image_folder='camera', root_dir='data/', images_num=2, transform=transform)
#%%
# Neuralt Network

#%%
# Training

#%%
# Testing

#%%
# Evaluation