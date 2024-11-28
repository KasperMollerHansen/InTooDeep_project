# %%
import torch
import warnings
# Suppress FutureWarning specifically from torch.load
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
torch.manual_seed(42)

### WARNING, THIS MIGHT REDUCE TRAINING ACCURACY ###
if torch.backends.cudnn.is_available()==True:
    print('CUDNN is available! ')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
####################################################

device = ("cuda" if torch.cuda.is_available() else "cpu")
try:
    import torch_directml
    device = torch_directml.device()
    print("DirectML is available, using DirectML")
except:
    print("DirectML is not available, using CPU/GPU")


import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
filename = os.path.basename(__file__)
model_name = root_dir+"/models/"+filename.split(".")[0]+".pth"
# Set the path to the root directory
sys.path.append(root_dir)
import windturbine as wt
import networks as nw

#%%
# Variables
############################################
def transform(image):
    # Convert image to array
    #image = np.array(image)
    #image = image[125:625,390:890]

    # Convert back to PIL Image for further transforms
    #image = Image.fromarray(image)

    # Compose transformations
    transform = transforms.Compose([
        transforms.CenterCrop(720),
        transforms.Resize(350),
        transforms.ToTensor(),  # Convert to tensor
    ])
    return transform(image)

angle_type = "both"
batch_size = 32
images_num = 1
base_angle_range = [300,60] # [0, 360] for all angles
model = nw.ResNet50_fm
############################################

wind_dataset = wt.WindTurbineDataset(csv_file='rotations_w_images.csv', image_folder='camera', 
                                     root_dir=root_dir+'/data/', images_num=images_num, transform=transform, angle_type=angle_type, base_angle_range=base_angle_range)
print(f"Dataset size: {len(wind_dataset)}")

train_dataset, test_dataset = wt.WindTurbineDataloader.train_test_split(wind_dataset, test_size=0.2)
trainloader = wt.WindTurbineDataloader.dataloader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = wt.WindTurbineDataloader.dataloader(test_dataset, batch_size=batch_size, shuffle=True)

try:
    model.load_state_dict(torch.load(model_name, map_location=torch.device("cpu")))
    print("Model loaded successfully")
except:
    print("Model not found, training from scratch")
model = model.to(device)

criterion = wt.AngularVectorLoss()
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)
#optimizer_2 = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0, dampening=0, weight_decay=0)

#Scheduler
schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, threshold=0.0001)

#Trainer
epochs = 20
accu_th = [20,10,5]
trainer = wt.Trainer(model, trainloader, testloader, criterion, optimizer,device, 
                     epochs=epochs, accu_th=accu_th, angle_type=angle_type, 
                     schedular=schedular, minimal=False)

# %%
model = trainer.train_model()

# Plot the training and testing loss
plt.figure(figsize=(10, 5))
# Make subplot with loss and accuracy
plt.subplot(1, 2, 1)
plt.plot(trainer.train_loss, label="Train Loss")
plt.plot(trainer.test_loss, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
if not trainer.minimal:
    # Make subplot with loss and accuracy
    plt.subplot(1, 2, 2)
    plt.plot(trainer.train_accuracy, label="Train Accuracy")
    plt.plot(trainer.test_accuracy, label="Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
plt.show()

# %%
# Save the model
torch.save(model.state_dict(), model_name)
print("Model saved successfully")

# %%
# Test the model
results = trainer.test_model(model.to(device), wind_dataset, angle_type=angle_type)
# Sort the results by base angle
results_sorted = results.sort_values(by="Base_Angle")

# Plot the results
if angle_type == "both" or angle_type == "base_angle":
    plt.figure(figsize=(10, 3))
    plt.stem(results_sorted["Base_Angle"], results_sorted["Base_Angle_Error"], label="Base Angle Error")
    plt.xlabel("Base Angle")
    plt.ylabel("Error")
    plt.legend()
    plt.grid()
    plt.show()
if angle_type == "both" or angle_type == "blade_angle":
    plt.figure(figsize=(10, 3))
    plt.stem(results_sorted["Blade_Angle"], results_sorted["Blade_Angle_Error"], label="Blade Angle Error")
    plt.xlabel("Blade Angle")
    plt.ylabel("Error")
    plt.legend()
    plt.grid()
    plt.show()

# %%
