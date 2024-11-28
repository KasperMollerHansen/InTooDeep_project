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
    change_csv_file("/data/rotations.csv") # Fix for jupyter

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
        angle_type (string, optional): Type of angle to predict. Default is "both". Options: "base_angle", "blade_angle", "both"
        base_angle_range (list, optional): Range of base angles to include. Default is [0, 360]. [min, max]
    """
    def __init__(self, csv_file, image_folder, root_dir, transform=None, images_num=1, angle_type="both", base_angle_range=[0, 360]):
        self.root_dir = root_dir
        csv_path = os.path.join(self.root_dir, csv_file)
        self.rotations_df = pd.read_csv(csv_path)
        self.images_num = images_num
        self.angle_type = angle_type
        self.transform = transform  # Accept external transformation callable
        self.image_folder = [
            os.path.join(image_folder + f"_0{i + 1}/") for i in range(images_num)
        ]
        # Filter out base angles outside the range
        if base_angle_range[1] > base_angle_range[0]:
            self.rotations_df = self.rotations_df[
                self.rotations_df.iloc[:, 0].apply(
                    lambda x: (x <= base_angle_range[1] and x >= base_angle_range[0])
                )
            ]
        else:
            self.rotations_df = self.rotations_df[
                self.rotations_df.iloc[:, 0].apply(
                    lambda x: (x <= base_angle_range[1] or x >= base_angle_range[0])
                )
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
        
        if self.angle_type == "base_angle":
            angles = torch.tensor([base_angle], dtype=torch.float32)
        else:
            angles = torch.tensor([base_angle, blade_angle], dtype=torch.float32)

        # Apply transformations
        images = [
            self.transform(image) if self.transform else self._default_transform(image)
            for image in images
        ]
        images = torch.cat(images, dim=0)
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
    def dataloader(dataset, batch_size=4, shuffle=True,):
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

#Loss functions
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
    
class AngularVectorLoss(nn.Module):
    def __init__(self):
        super(AngularVectorLoss, self).__init__()
    
    def forward(self, pred_init, target_init, is_degrees=False):
        # Test if the dimensions are correct
        if pred_init.shape != target_init.shape:
            raise ValueError("The dimensions of the predicted and target tensors must match.")
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
        loss = []
        for i in range(pred.shape[1]):
            # Represent angles as 2D vectors
            scale = 1+(i*2)
            pred_vec = torch.stack((torch.sin(pred[:,i]*scale), torch.cos(pred[:,i]*scale)), dim=-1)
            target_vec = torch.stack((torch.sin(target[:,i]*scale), torch.cos(target[:,i]*scale)), dim=-1)

            # Compute vector difference and its norm
            diff = pred_vec - target_vec
            diff = diff * 10  # Scale by a factor of 10
            loss.append(torch.mean(torch.norm(diff, dim=-1) ** 2))  # Mean squared norm of differences
        
        return torch.sum(torch.stack(loss))

# Trainers
class Trainer():
    train_loss = []
    test_loss = []

    def __init__(self, model, trainloader, testloader, criterion, optimizer,
                  device, epochs, accu_th, angle_type, schedular=None, minimal=False):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.accu_th = accu_th if isinstance(accu_th, list) else [accu_th]
        self.angle_type = angle_type
        self.schedular = schedular
        self.minimal = minimal
    
    def _accuracy_angle(self, pred, target, threshold, is_degrees=True):
        pred = pred.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        accu_list = []
        for i in range(pred.shape[1]):
            diff = np.abs(pred[:, i] - target[:, i])
            if is_degrees:
                val = 360/(1+i*2)
                diff = np.fmod(diff, val)
                diff = np.minimum(diff, val - diff)
            else:
                val = 2*np.pi/(1+i*2)
                diff = np.fmod(diff, val)
                diff = np.minimum(diff, val - diff)

            for j in range (len(threshold)):
                within_threshold = np.mean(diff <= threshold[j])
                accuracy = within_threshold * 100
                accu_list.append(accuracy)
        return np.array(accu_list)
    
    def _train(self, dataloader, model, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0

        for _, data in enumerate(dataloader):
            inputs, labels = data
            # Only get the first label
            inputs, labels = inputs.to(device), labels.to(device)

            # Compute the prediction error
            pred = model(inputs)
            loss = criterion(pred, labels, is_degrees=True)
            running_loss += loss.item()
            # Accuracy
            if not self.minimal:
                accuracy = self._accuracy_angle(pred, labels, self.accu_th, is_degrees=True)
                running_accuracy += accuracy

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if self.schedular:
            self.schedular.step(loss)

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
                labels = labels
                inputs, labels = inputs.to(device), labels.to(device)

                # Compute the prediction error
                pred = model(inputs)
                loss = criterion(pred, labels, is_degrees=True)
                running_loss += loss.item()
                # Accuracy
                if not self.minimal:
                    accuracy = self._accuracy_angle(pred, labels, self.accu_th, is_degrees=True)
                    running_accuracy += accuracy

        avg_loss = running_loss / len(dataloader)
        avg_accuracy = running_accuracy / len(dataloader)
        return avg_loss, avg_accuracy
    
    def train_model(self):
        for epoch in range(self.epochs):
            train_loss, train_acc = self._train(self.trainloader, self.model, self.criterion, self.optimizer, self.device)
            self.train_loss.append(train_loss)
            test_loss, test_acc = self._test(self.testloader, self.model, self.criterion, self.device)
            self.test_loss.append(test_loss)
            print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {np.round(train_loss,3)}, Test Loss: {np.round(test_loss,3)}")
            print(f"lr {self.schedular.get_last_lr}")
            
            if not self.minimal:
                try:
                    self.train_accuracy = np.vstack([self.train_accuracy, train_acc])
                    self.test_accuracy = np.vstack([self.test_accuracy, test_acc])
                except:
                    self.train_accuracy = train_acc
                    self.test_accuracy = test_acc
                    accu_th_x = self.accu_th*int(len(train_acc)/len(self.accu_th))

                table = [accu_th_x,np.round(train_acc,2).tolist(), np.round(test_acc,2).tolist()]
                df = pd.DataFrame(table, index=["Angle","Train Accuracy", "Test Accuracy"])
                print(df.to_string(header=False))
        return self.model
    
    def test_model(self, model, dataset, angle_type="base_angle"):
        batch_size = 50
        dataloader = WindTurbineDataloader.dataloader(dataset, batch_size=batch_size, shuffle=False)
        model.eval()
        
        # Create a list to accumulate results
        results_list = []
        
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Compute the prediction
                pred = model(inputs)

                if angle_type == "base_angle":
                    # Collect results in a list
                    for j in range(len(labels)):
                        results_list.append({
                            "Image": f"image_01_{i*batch_size+j +1}",
                            "Base_Angle": labels[j].item(),
                            "Base_Angle_Pred": pred[j].item()

                        })
                else:
                    # Collect results in a list
                    for j in range(len(labels)):
                        results_list.append({
                            "Image": f"image_01_{i*batch_size+j +1}",
                            "Base_Angle": labels[j,0].item(),
                            "Base_Angle_Pred": pred[j,0].item(),
                            "Blade_Angle": labels[j,1].item(),
                            "Blade_Angle_Pred": pred[j,1].item()
                        })
            
            # Convert the list of results into a DataFrame
            results = pd.DataFrame(results_list)
            # If results contain a column named 'Base_angle', do the following
            if "Base_Angle" in results.columns:
                # Normalize 'Base_Angle_Pred' to the range [0, 360)
                results["Base_Angle_Pred_Pos"] = np.mod(results["Base_Angle_Pred"], 360)

                # Calculate the 'Base_Angle_Error' as the difference between the predicted and actual angles
                results["Base_Angle_Error"] = results["Base_Angle_Pred_Pos"] - results["Base_Angle"]

                # Apply the wrapping logic in a vectorized manner using numpy's where
                results["Base_Angle_Error"] = np.where(
                    results["Base_Angle_Error"] > 180, 
                    results["Base_Angle_Error"] - 360, 
                    np.where(results["Base_Angle_Error"] < -180, 
                            results["Base_Angle_Error"] + 360, 
                            results["Base_Angle_Error"])
                )
            # If results contain a column named 'Blade_angle', do the following
            if "Blade_Angle" in results.columns:
                # Normalize 'Blade_Angle_Pred' to the range [0, 120)
                results["Blade_Angle_Pred_Pos"] = np.mod(results["Blade_Angle_Pred"], 120)

                # Calculate the 'Blade_Angle_Error' as the difference between the predicted and actual angles
                results["Blade_Angle_Error"] = results["Blade_Angle_Pred_Pos"] - results["Blade_Angle"]

                # Apply the wrapping logic in a vectorized manner using numpy's where
                results["Blade_Angle_Error"] = np.where(
                    results["Blade_Angle_Error"] > 60, 
                    results["Blade_Angle_Error"] - 120, 
                    np.where(results["Blade_Angle_Error"] < -60, 
                            results["Blade_Angle_Error"] + 120, 
                            results["Blade_Angle_Error"])
                )
            
            return results