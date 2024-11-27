# %%
from torch import nn
from torchsummary import summary
import torch
#%%
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
    
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_Regressor_4_conv().to(device)
    # Print summary for a (3, 300, 300) input
    summary(model, input_size=(3, 300, 300), device=device.type)
#%%
class CNN_test_2_output(nn.Module):
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
            nn.Linear(in_features=128, out_features=2)  # Output one angle
        )

    def forward(self, x):
        x = self.convolution_stack(x)
        x = self.linear_stack(x)
        return x
    
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_test_2_output().to(device)
    # Print summary for a (3, 300, 300) input
    summary(model, input_size=(3, 300, 300), device=device.type)
# %%
ResNet34 = torch.hub.load('pytorch/vision:v0.20.0', 'resnet34')
ResNet34.fc = torch.nn.Linear(in_features=512,out_features=1,bias=True)
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet34.to(device)
    # Print summary for a (3, 300, 300) input
    summary(model, input_size=(3, 300, 300), device=device.type)
    
#%%

class CNN_Reg(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*37*37, 128),  # Adjust dimensions for 300x300 input
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.convolution_stack(x)
        x = self.linear_stack(x)
        return x
    
    
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_Reg().to(device)
    # Print summary for a (3, 300, 300) input
    summary(model, input_size=(3, 300, 300), device=device.type)
     
#%%