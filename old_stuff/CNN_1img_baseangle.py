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
    def __init__(self, csv_file, image_folder, root_dir, images_num=1, transform_size=None, grayscale=False):
        self.root_dir = root_dir
        csv_path =  os.path.join(self.root_dir, csv_file)
        self.rotations_df = pd.read_csv(csv_path)
        self.transform_size = transform_size
        self.images_num = images_num
        self.image_folder = []
        self.grayscale = grayscale
        for i in range(images_num):
            self.image_folder.append(image_folder+f"_0{i+1}/")

    def __len__(self):
        return len(self.rotations_df)
    
    def _transform(self, transform_size, image, grayscale=False):
        # Convert image to array
        image = np.array(image)
        image = image[225:525,490:790]
        if grayscale:
            # Convert to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Convert back to PIL Image for further transforms
        image = Image.fromarray(image)

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

        images = [self._transform(self.transform_size, image, self.grayscale) for image in images]
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
class CNN_Regressor_4_conv(nn.Module):
    def __init__(self):
        super().__init__()
        # Input image size: (300, 300, 3)
        self.convolution_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=7, stride=1, padding=0),  # (294, 294, 16)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (147, 147, 16)
            nn.Dropout(p=0.2),  # Dropout to reduce overfitting

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0),  # (143, 143, 32)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (71, 71, 32)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),  # (69, 69, 64)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (34, 34, 64)

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),  # (32, 32, 128)
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((5, 5))  # (5, 5, 128)
        )

        self.linear_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=5 * 5 * 128, out_features=128),  # Adjusted input size
            nn.ReLU(),
            nn.Dropout(p=0.3),  # Dropout to reduce overfitting
            nn.Linear(in_features=128, out_features=1)  # Output one angle
        )

    def forward(self, x):
        x = self.convolution_stack(x)
        x = self.linear_stack(x)
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_Regressor_4_conv().to(device)

# Print summary for a (3, 300, 300) input
summary(model, input_size=(3, 300, 300), device=device.type)

#%%
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
        
        # Scale the loss by a factor of 10
        angular_diff = angular_diff * 10
        
        # Loss is the mean squared angular difference
        loss = torch.mean((angular_diff)** 2)
        return loss


# %%
# Load data
wind_dataset = WindTurbineDataset(csv_file='rotations_w_images.csv', image_folder='camera', root_dir='data/', images_num=1, grayscale=False)

train_dataset, test_dataset = WindTurbineDataloader.train_test_split(wind_dataset, test_size=0.2)
trainloader = WindTurbineDataloader.dataloader(train_dataset, batch_size=16*2, shuffle=True)
testloader = WindTurbineDataloader.dataloader(test_dataset, batch_size=16*2, shuffle=True)
# Load model
model = CNN_Regressor_4_conv().to(device)
model.load_state_dict(torch.load("models/CNN_Regressor_4_conv.pth"))
# Loss function
criterion = AngularLoss()
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# %%
# Training
class Trainer:
    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []


    def __init__(self, model, trainloader, testloader, criterion, optimizer, device, epochs=10):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
    
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
            accuracy = self._accuracy_angle(pred, labels, 20, is_degrees=True)
            running_accuracy += accuracy

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        schedular.step()
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
                accuracy = self._accuracy_angle(pred, labels, 10, is_degrees=True)
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


#%%
trainer = Trainer(model, trainloader, testloader, criterion, optimizer, device, epochs=20)
trainer.train_model()
# Plot the training and testing loss
plt.figure(figsize=(10, 5))
# Make subplot with loss and accuracy
plt.subplot(1, 2, 1)
plt.plot(trainer.train_accuracy, label="Train Accuracy")
plt.plot(trainer.test_accuracy, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
# Make subplot with loss and accuracy
plt.subplot(1, 2, 2)
plt.plot(trainer.train_loss, label="Train Loss")
plt.plot(trainer.test_loss, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()
#%%
# Save the model
torch.save(model.state_dict(), "models/CNN_Regressor_4_conv.pth")


#%%
# Plot 3 images in row
fig, axs = plt.subplots(1, 3, figsize=(15, 15))
for i in range(3):
    image, angles = wind_dataset[i+1756]
    image = image.numpy().transpose((1, 2, 0))
    axs[i].imshow(image)
    axs[i].set_title(f"Base Angle: {angles[0]:.2f}, Blade Angle: {angles[1]:.2f}")
    axs[i].axis('off')
plt.axis('off')
plt.show()

#%%
# %%
class CNN_Regressor_gray(nn.Module):
    def __init__(self):
        super().__init__()
        # Input image (300,300,1)
        self.convolution_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=7, stride=1, padding=0),  # (294, 294, 20)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),  # Dropout to reduce overfitting

            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5, stride=1, padding=0),  # (290, 290, 40)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (145, 145, 40)

            nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3, stride=1, padding=0),  # (143, 143, 40)
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=40, out_channels=60, kernel_size=3, stride=2, padding=0),  # (71, 71, 60)
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),  # Dropout to reduce overfitting

            nn.Conv2d(in_channels=60, out_channels=80, kernel_size=3, stride=2, padding=0),  # (35, 35, 80)
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=80, out_channels=100, kernel_size=3, stride=2, padding=0),  # (17, 17, 100)
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=100, out_channels=120, kernel_size=3, stride=2, padding=0),  # (8, 8, 120)
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=120, out_channels=140, kernel_size=3, stride=2, padding=0),  # (3, 3, 140)
            nn.ReLU(inplace=True),
        )

        self.linear_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=3*3*140, out_features=128),  # Adjusted input size
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # Dropout to reduce overfitting
            nn.Linear(in_features=32, out_features=1)  # Output one angle
        )

    def forward(self, x):
        x = self.convolution_stack(x)
        x = self.linear_stack(x)
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_Regressor_gray().to(device)

# Print summary for a (1, 300, 300) input
summary(model, input_size=(1, 300, 300), device=device.type)

#%%
# Evaluation