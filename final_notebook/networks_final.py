# %%
from torch import nn
from torchsummary import summary
import torch
import torchvision
#%%
# %%
ResNet34 = torchvision.models.resnet34(weights=False)
ResNet34.fc = torch.nn.Linear(in_features=512,out_features=2,bias=True)
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet34.to(device)
    # Print summary for a (3, 300, 300) input
    summary(model, input_size=(3, 300, 300), device=device.type)
