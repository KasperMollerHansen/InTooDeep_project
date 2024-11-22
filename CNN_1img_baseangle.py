# %%
# Load packages
import os
import cv2
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
        self.images_num = images_num
        self.image_folder = []
        for i in range(images_num):
            self.image_folder.append(image_folder+f"_0{i+1}/")

    def __len__(self):
        return len(self.rotations_df)
    
    def _transform(self, transform_size, image):
        # Convert image to tensor first to determine its shape
        to_tensor = transforms.ToTensor()
        tensor_image = to_tensor(image)
        
        # Dynamically calculate mean and std based on the number of channels
        num_channels = tensor_image.shape[0]
        mean = [0.5 for _ in range(num_channels)]
        std = [0.5 for _ in range(num_channels)]
        normalize = transforms.Normalize(mean=mean, std=std)

        # Apply contrast and brightness adjustment using cv2
        alpha = 1.5  # Contrast control (1.0-3.0)
        beta = 10     # Brightness control (0-100)
        
        # Convert image to numpy array for cv2 operations
        img_cv = np.array(image)
        
        # Apply contrast and brightness adjustment
        img_adjusted = cv2.convertScaleAbs(img_cv, alpha=alpha, beta=beta)

        # Convert back to PIL Image for further transforms
        img_adjusted = Image.fromarray(img_adjusted)

        # Determine resize size
        if transform_size is None:
            x, y = image.size
            size_transform = transforms.Resize((int(y), int(x)))
        else:
            size_transform = transforms.Resize((int(transform_size[0]), int(transform_size[1])))

        # Compose transformations
        transform = transforms.Compose([
            size_transform,
            transforms.ToTensor(),  # Convert to tensor
            #normalize               # Normalize with dynamically calculated mean and std
        ])
        return transform(img_adjusted)
    

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
    

# %%
class CNN_Regressor_4(nn.Module):
    def __init__(self):
        super().__init__()
        # Original Image (720, 1280, 6) -> Downscaled by factor 4 -> Input image (180, 320, 6)
        self.convolution_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=3),  # (180, 320, 16)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (90, 160, 16)

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),  # (90, 160, 32)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (45, 80, 32)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # (45, 80, 64)
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((5, 5))  # (5, 5, 64)
        )

        self.linear_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=5 * 5 * 64, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # Dropout to reduce overfitting
            nn.Linear(in_features=128, out_features=1)  # Output one angle
        )

    def forward(self, x):
        x = self.convolution_stack(x)
        x = self.linear_stack(x)
        return x

model = CNN_Regressor_4().to(device)
summary(model, input_size = (3, 180, 320), device=device)

# Loss function
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
        
        # Loss is the mean squared angular difference
        loss = torch.mean(angular_diff ** 2)
        return loss


# %%
# Load data
wind_dataset = WindTurbineDataset(csv_file='rotations_w_images.csv', image_folder='camera', root_dir='data/', images_num=1, transform_size=np.array([720,1280])/2)
train_dataset, test_dataset = WindTurbineDataloader.train_test_split(wind_dataset, test_size=0.2)
trainloader = WindTurbineDataloader.dataloader(train_dataset, batch_size=16, shuffle=True)
testloader = WindTurbineDataloader.dataloader(test_dataset, batch_size=16, shuffle=True)
# Load model
model = CNN_Regressor_4().to(device)
# Loss function
criterion = AngularLoss()
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# %%
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

        for i, data in enumerate(dataloader):
            inputs, labels = data
            # Only get the first label
            labels = labels[:, 0].flatten()
            inputs, labels = inputs.to(device), labels.to(device)

            # Compute the prediction error
            pred = model(inputs).flatten()
            loss = criterion(pred, labels, is_degrees=True)
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
                labels = labels[:, 0].flatten()
                inputs, labels = inputs.to(device), labels.to(device)

                # Compute the prediction error
                pred = model(inputs).flatten()
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


# %%
# Small trainer

train_size = 20
small_dataset, _ = torch.utils.data.random_split(wind_dataset, [train_size, len(wind_dataset) - train_size])

# Adjust the batch size if needed
batch_size = 8  # Increase the batch size if the dataset size is larger

# Create a DataLoader for the small dataset
small_loader = WindTurbineDataloader.dataloader(small_dataset, batch_size=batch_size, shuffle=True)

# Train the model with the slightly larger dataset
small_trainer = Trainer(model, small_loader, small_loader, criterion, optimizer, device, epochs=50)
small_trainer.train_model()


# Plot the training and testing loss
plt.plot(small_trainer.train_loss, label="Train Loss")
plt.plot(small_trainer.test_loss, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

#%%
trainer = Trainer(model, trainloader, testloader, criterion, optimizer, device, epochs=20)
trainer.train_model()
# Plot the training and testing loss
plt.plot(trainer.train_loss, label="Train Loss")
plt.plot(trainer.test_loss, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()
#%%
# Testing

#%%
# Evaluation