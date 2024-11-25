# %%
import torch
import warnings
# Suppress FutureWarning specifically from torch.load
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

device = ("cuda" if torch.cuda.is_available() else "cpu")

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
def transform(image):
    # Convert image to array
    image = np.array(image)
    image = image[225:525,490:790]

    # Convert back to PIL Image for further transforms
    image = Image.fromarray(image)

    # Compose transformations
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])
    return transform(image)

wind_dataset = wt.WindTurbineDataset(csv_file='rotations_w_images.csv', image_folder='camera', root_dir=root_dir+'/data/', images_num=1, transform=transform)

train_dataset, test_dataset = wt.WindTurbineDataloader.train_test_split(wind_dataset, test_size=0.2)
trainloader = wt.WindTurbineDataloader.dataloader(train_dataset, batch_size=16, shuffle=True)
testloader = wt.WindTurbineDataloader.dataloader(test_dataset, batch_size=16, shuffle=True)

model = nw.CNN_Regressor_4_conv().to(device)
try:
    model.load_state_dict(torch.load(model_name))
    print("Model loaded successfully")
except:
    print("Model not found, training from scratch")

criterion = wt.AngularLoss()
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

# %%
trainer = wt.Trainer_base_angle(model, trainloader, testloader, criterion, optimizer, device, accu_th=20, epochs=40)
model = trainer.train_model()

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

# %%
# Save the model
torch.save(model.state_dict(), model_name)
print("Model saved successfully")

# %%
