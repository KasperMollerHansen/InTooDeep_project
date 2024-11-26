# %%
import torch
import warnings
# Suppress FutureWarning specifically from torch.load
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


### WARNING, THIS MIGHT REDUCE TRAINING ACCURACY ###
torch.set_float32_matmul_precision("highest")
if torch.backends.cudnn.is_available()==True:
    print('CUDNN is available! ')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
####################################################

device = ("cuda" if torch.cuda.is_available() else "cpu")

import sys
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
filename = os.path.basename(__file__)
model_name = root_dir+"/models/"+filename.split(".")[0]+".pth"
# Set the path to the root directory
sys.path.append(root_dir)
import windturbine_CUDA as wt
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

batch_size = 64

wind_dataset = wt.WindTurbineDataset(csv_file='rotations_w_images.csv', image_folder='camera', root_dir=root_dir+'/data/', images_num=2, transform=transform)

train_dataset, test_dataset = wt.WindTurbineDataloader.train_test_split(wind_dataset, test_size=0.2)
trainloader = wt.WindTurbineDataloader.dataloader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True)
testloader = wt.WindTurbineDataloader.dataloader(test_dataset, batch_size=batch_size, shuffle=True,pin_memory=False)

model = nw.ResNet34_fm.to(device)

try:
    model.load_state_dict(torch.load(model_name))
    print("Model loaded successfully")
except:
    print("Model not found, training from scratch")

criterion = wt.AngularVectorLoss()
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0000001)
schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, threshold=0.0001)

# %%
trainer = wt.Trainer_base_angle(model, trainloader, testloader, criterion, optimizer,
                                 device, epochs=10, accu_th=[20,10,5], schedular=schedular, minimal=False)
print("Allocated:", round(torch.cuda.memory_allocated(0)/(10243*(10**3)),1), "GB")
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

'''
todo: add evaluation program to predict each test data in one big
dataframe. Then find a way to access images within a range of rotation.
'''



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
im1 = inputs[:,3:6,:,:]
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
