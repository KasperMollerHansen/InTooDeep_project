#%%
import os
import sys
import torch
import pandas as pd
import numpy as np
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

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

if __name__ == "__main__":
    change_csv_file("data/rotations.csv")

# Dataset
class WindTurbineDataset(Dataset):
    """
    A dataset class for wind turbine images.

    Args:
        root_dir (string): Directory with csv and image folder.
        csv_file (string): Path to the csv file with filenames and angles.
        image_folder (string): Directory with images.
        transform (callable, optional): Optional transform to be applied on a sample.
        images_num (int, optional): Number of images per sample. Default is 1.
        grayscale (bool, optional): Whether to convert images to grayscale. Default is False.
    """
    def __init__(self, csv_file, image_folder, root_dir, transform=None, images_num=1):
        self.root_dir = root_dir
        csv_path = os.path.join(self.root_dir, csv_file)
        self.rotations_df = pd.read_csv(csv_path)
        self.images_num = images_num
        self.transform = transform  # Accept external transformation callable
        self.image_folder = [
            os.path.join(image_folder + f"_0{i + 1}/") for i in range(images_num)
        ]

    def __len__(self):
        return len(self.rotations_df)
    
    def _default_transform(self, image):
        # Compose transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transform(image)

    def __getitem__(self, idx):
        # Get the image file paths
        images = []
        for i in range(self.images_num):
            img_name = os.path.join(
                self.root_dir, self.image_folder[i], self.rotations_df.iloc[idx, i + 2])  # 'filename' is the third column
            images.append(Image.open(img_name))

        # Retrieve angles
        base_angle = self.rotations_df.iloc[idx, 0]
        blade_angle = self.rotations_df.iloc[idx, 1]
        angles = torch.tensor([base_angle, blade_angle], dtype=torch.float32)

        # Apply transformations
        images = [
            self.transform(image) if self.transform else self._default_transform(image)
            for image in images
        ]
        images = torch.cat(images, dim=0)  # Concatenate along the channel dimension
        return images, angles
    
# Dataloader
class WindTurbineDataloader(Dataset):
    @staticmethod
    def train_test_split(dataset, test_size=0.2):
        # Split the dataset into training and testing sets
        train_size = int((1 - test_size) * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        # Seperate the labels from the features
        return train_dataset, test_dataset
    @staticmethod
    def dataloader(dataset, batch_size=4, shuffle=True):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class AngularLoss(nn.Module):
    def __init__(self):
        super(AngularLoss, self).__init__()
    
    def forward(self, pred_init, target_init, is_degrees=False):
        pred = pred_init
        target = target_init
        """
        Computes the loss for angular data.
        pred: Tensor of shape (batch_size, 2), predicted angles (in degrees or radians).
        target: Tensor of shape (batch_size, 2), target angles (in degrees or radians).
        is_degrees: Boolean indicating if the angles are in degrees (default: False).
        """
        if is_degrees:
            pred = pred * (torch.pi / 180)
            target = target * (torch.pi / 180)
        
        # Compute smallest angular difference
        angular_diff = torch.atan2(torch.sin(pred - target), torch.cos(pred - target))
        
        # Scale the loss by a factor of 10
        angular_diff = angular_diff * 10
        
        # Loss is the mean squared angular difference
        loss = torch.mean((angular_diff)** 2)
        return loss

# Training
class Trainer():
    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []

    def __init__(self, model, trainloader, testloader, criterion, optimizer, device, epochs, accu_th, schedular=None):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.accu_th = accu_th
        self.schedular = schedular
    
    def _accuracy_angle(self, pred, target, threshold, is_degrees=True):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()

        diff = np.abs(pred - target)
        if is_degrees:
            diff = np.minimum(diff, 360 - diff)
        else:
            diff = np.minimum(diff, 2*np.pi - diff)
        
        within_threshold = np.mean(diff <= threshold)
        accuracy = within_threshold * 100

        return accuracy
    
    def _train(self, dataloader, model, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0

        for i, data in enumerate(dataloader):
            inputs, labels = data
            # Only get the first label
            labels = labels[:, 0].flatten()
            inputs, labels = inputs.to(device), labels.to(device)

            # Compute the prediction error
            pred = model(inputs).flatten()
            loss = criterion(pred, labels, is_degrees=True)
            running_loss += loss.item()
            # Accuracy
            accuracy = self._accuracy_angle(pred, labels, self.accu_th, is_degrees=True)
            running_accuracy += accuracy

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if self.schedular:
            self.schedular.step()

        avg_loss = running_loss / len(dataloader)
        avg_accuracy = running_accuracy / len(dataloader)
        return avg_loss, avg_accuracy
    
    def _test(self, dataloader, model, criterion, device):
        model.eval()
        running_loss = 0.0
        running_accuracy = 0.0

        with torch.no_grad():
            for _, data in enumerate(dataloader):
                inputs, labels = data
                labels = labels[:, 0].flatten()
                inputs, labels = inputs.to(device), labels.to(device)

                # Compute the prediction error
                pred = model(inputs).flatten()
                loss = criterion(pred, labels, is_degrees=True)
                running_loss += loss.item()
                # Accuracy
                accuracy = self._accuracy_angle(pred, labels, self.accu_th, is_degrees=True)
                running_accuracy += accuracy

        avg_loss = running_loss / len(dataloader)
        avg_accuracy = running_accuracy / len(dataloader)
        return avg_loss, avg_accuracy
    
    def train_model(self):
        for epoch in range(self.epochs):
            train_loss, train_acc = self._train(self.trainloader, self.model, self.criterion, self.optimizer, self.device)
            self.train_loss.append(train_loss)
            self.train_accuracy.append(train_acc)
            test_loss, test_acc = self._test(self.testloader, self.model, self.criterion, self.device)
            self.test_loss.append(test_loss)
            self.test_accuracy.append(test_acc)
            print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {np.round(train_loss,3)}, Test Loss: {np.round(test_loss,3)}")
            print(f"Train Accuracy: {np.round(train_acc,3)} %, Test Accuracy: {np.round(test_acc,3)} %")
        return self.model

# %%
