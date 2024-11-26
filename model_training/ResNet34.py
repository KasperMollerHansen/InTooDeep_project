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

batch_size = 16*4

wind_dataset = wt.WindTurbineDataset(csv_file='rotations_w_images.csv', image_folder='camera', root_dir=root_dir+'/data/', images_num=1, transform=transform)

train_dataset, test_dataset = wt.WindTurbineDataloader.train_test_split(wind_dataset, test_size=0.2)
trainloader = wt.WindTurbineDataloader.dataloader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = wt.WindTurbineDataloader.dataloader(test_dataset, batch_size=batch_size, shuffle=True)

model = nw.ResNet34
try:
    model.load_state_dict(torch.load(model_name, map_location=torch.device("cpu")))
    print("Model loaded successfully")
except:
    print("Model not found, training from scratch")
model = model.to(device)

criterion = wt.AngularVectorLoss()
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# %%
trainer = wt.TrainerBaseAngle(model, trainloader, testloader, criterion, optimizer,
                                 device, epochs=1, accu_th=[20,10,5], schedular=schedular, minimal=False)
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
torch.save(model.to("cpu").state_dict(), model_name)
print("Model saved successfully")

# %%
# Test the model
results = trainer.test_model(model.to(device), wind_dataset)
# Sort the results by base angle
results_sorted = results.sort_values(by="Base_Angle")

# Plot the results
plt.figure(figsize=(10, 5))
plt.stem(results_sorted["Base_Angle"], results_sorted["Base_Angle_Error"], label="Base Angle Error")
plt.xlabel("Epochs")







# %%
# Old code
model.eval()
running_loss = 0.0

with torch.no_grad():
    for _, data in enumerate(testloader):
        inputs, labels = data
        inputs_n, labels_n = inputs.to(device), labels.to(device)

        # Compute the prediction error
        pred = model(inputs_n)
        loss = criterion(pred,labels_n,is_degrees=True)
        running_loss += loss.item()
avg_loss = running_loss / len(testloader)
# %%
im1 = inputs
im1 = torch.permute(im1,(0,2,3,1)).numpy()
preds = torch.Tensor.cpu(pred).numpy()
rot_base = labels[:,0]
#rot_wings = labels[:,1]
pred_base = preds[:,0]
#pred_wings = preds[:,1]
print(pred_base)

ax = plt.subplot(3,1,1)
ax.imshow(im1[0,:,:,:])
ax.set_axis_off()
ax.set_title(f"Pred: {pred_base[0]:.1f}, Actual: {rot_base[0]:.1f}")

ax = plt.subplot(3,1,2)
ax.imshow(im1[1,:,:,:])
ax.set_axis_off()
ax.set_title(f"Pred: {pred_base[1]:.1f}, Actual: {rot_base[1]:.1f}")


ax = plt.subplot(3,1,3)
ax.imshow(im1[2,:,:,:])
ax.set_axis_off()
ax.set_title(f"Pred:{pred_base[2]:.1f}, Actual: {rot_base[2]:.1f}")

plt.tight_layout()
# %%
