# %%
# Load packages
import os
import sklearn
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torchsummary import summary

device = ("cuda" if torch.cuda.is_available() else "cpu")

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

# Dataset
class WindTurbineDataset(Dataset):
    """
        Args:
            root_dir (string): Directory with csv and image folder.
            csv_file (string): Path to the csv file with filenames and angles
            image_folder (string): Directory with images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
    def __init__(self, csv_file, image_folder, root_dir, images_num=1, transform_size=None):
        self.root_dir = root_dir
        csv_path =  os.path.join(self.root_dir, csv_file)
        self.rotations_df = pd.read_csv(csv_path)
        self.transform_size = transform_size
        self.image_folder = []
        for i in range(images_num):
            self.image_folder.append(image_folder+f"_0{i+1}/")

    def __len__(self):
        return len(self.rotations_df)
    
    def _transform(self, transform_size, image):
    
        if transform_size.any() == None:
            #Get size of the image
            x,y = image.size
            transform = transforms.Compose([
            transforms.Resize((int(y),int(x))), # Resizing the image to 360x640
            transforms.ToTensor()
            ])
        else:
            transform = transforms.Compose([
            transforms.Resize((int(transform_size[0]),int(transform_size[1]))), # Resizing the image to 360x640
            transforms.ToTensor()
        ])
        return transform(image)


    def __getitem__(self, idx):
        # Get the image file path
        images = []
        for i in range(len(self.image_folder)):
            img_name = (os.path.join(self.root_dir, self.image_folder[i], self.rotations_df.iloc[idx, i+2])) #'filename' is the third column
            images.append(Image.open(img_name))
        base_angles = self.rotations_df.iloc[idx, 0] 
        blade_angles = self.rotations_df.iloc[idx, 1]
        angles = torch.tensor([base_angles, blade_angles], dtype=torch.float32)

        images = [self._transform(self.transform_size, image) for image in images]
        images = torch.concatenate(images, dim=0)  

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
#%%
# Neuralt Network
class CNN_Regressor_4(nn.Module):
    def __init__(self):
        super().__init__()

        # Original Image (720, 1280, 6) -> Downscaled by factor 4 -> Input image (180, 320, 6)
        
        self.convolution_stack = nn.Sequential(
            nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size = 15, stride = 1, bias = True), # Size (164, 306, 12)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=6, stride=6), # Size (27, 51, 12)
            
            nn.Conv2d(in_channels = 12, out_channels = 24, kernel_size = 7, stride = 1, bias = True), # Size (21, 45, 24)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = 6, stride = 6), # Size (3, 7, 24)
        )
        
        self.linear_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 3*7*24, out_features = 128, bias = True),
            nn.ReLU(),
            nn.Linear(in_features = 128, out_features = 2, bias = True)
        )
    
    def forward(self, x):
        conv_out = self.convolution_stack(x)
        fc_out = self.linear_stack(conv_out)
        return fc_out

model = CNN_Regressor_4().to(device)
summary(model, input_size = (6, 180, 320), device=device)

# Loss function
class DualAngularLoss(nn.Module):
    def __init__(self):
        super(DualAngularLoss, self).__init__()
    
    def forward(self, pred, target, is_degrees=False):
        """
        pred: Tensor of shape (batch_size, 2), predicted angles (can be in degrees or radians)
        target: Tensor of shape (batch_size, 2), target angles (can be in degrees or radians)
        is_degrees: Boolean indicating if the input angles are in degrees (default: False)
        """
        # Convert degrees to radians if needed
        if is_degrees:
            pred_rad = pred * (torch.pi / 180)
            target_rad = target * (torch.pi / 180)
        else:
            pred_rad = pred
            target_rad = target

        # Compute angular difference for both angles (in radians)
        diff1 = 1 - torch.cos(pred_rad[:, 0] - target_rad[:, 0])
        diff2 = 1 - torch.cos(pred_rad[:, 1] - target_rad[:, 1])
        
        # Mean over the batch
        loss = torch.mean(diff1**2 + diff2**2)
        return loss

#%%
# Load data
wind_dataset = WindTurbineDataset(csv_file='rotations_w_images.csv', image_folder='camera', root_dir='data/', images_num=2, transform_size=np.array([720,1280])/4)
train_dataset, test_dataset = WindTurbineDataloader.train_test_split(wind_dataset, test_size=0.2)
trainloader = WindTurbineDataloader.dataloader(train_dataset, batch_size=8, shuffle=True)
testloader = WindTurbineDataloader.dataloader(test_dataset, batch_size=8, shuffle=True)
# Load model
model = CNN_Regressor_4().to(device)
# Loss function
criterion = DualAngularLoss()
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
#%%
# Training
class Trainer:
    train_loss = []
    test_loss = []

    def __init__(self, model, trainloader, testloader, criterion, optimizer, device, epochs=10):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
    
    def _train(self, dataloader, model, criterion, optimizer, device):
        model.train()
        running_loss = 0.0

        for _, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Compute the prediction error
            pred = model(inputs)
            loss = criterion(pred[0], labels[0], is_degrees=True)
            running_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        schedular.step()
        avg_loss = running_loss / len(dataloader)
        return avg_loss
    
    def _test(self, dataloader, model, criterion, device):
        model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for _, data in enumerate(dataloader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # Compute the prediction error
                pred = model(inputs)
                loss = criterion(pred, labels, is_degrees=True)
                running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        return avg_loss
    
    def train_model(self):
        for epoch in range(self.epochs):
            train_loss = self._train(self.trainloader, self.model, self.criterion, self.optimizer, self.device)
            test_loss = self._test(self.testloader, self.model, self.criterion, self.device)
            self.train_loss.append(train_loss)
            self.test_loss.append(test_loss)
            print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {train_loss}, Test Loss: {test_loss}")
#%%
trainer = Trainer(model, trainloader, testloader, criterion, optimizer, device, epochs=10)
trainer.train_model()
# Plot the training and testing loss
plt.plot(trainer.train_loss, label="Train Loss")
plt.plot(trainer.test_loss, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


#%%
# Testing

#%%
# Evaluation