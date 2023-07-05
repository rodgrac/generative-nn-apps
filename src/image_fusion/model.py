import torch
import torch.nn as nn

from torchvision import models


class BimodalAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc1 = FeatureEncoder()
        self.enc2 = FeatureEncoder()
        
        self.lnorm1 = nn.LayerNorm([64, 32, 32])
        self.lnorm2 = nn.LayerNorm([64, 32, 32])
        
        self.fuseconv = nn.Conv2d(128, 64, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        
        self.dec1 = FeatureDecoder()
        self.dec2 = FeatureDecoder()
        
    
    def forward(self, x1, x2):
        x1 = self.enc1(x1)
        x1 = self.lnorm1(x1)
        
        x2 = self.enc2(x2)
        x2 = self.lnorm2(x2)
        
        x = torch.cat((x1, x2), dim=1)
        
        x = self.fuseconv(x)
        x = self.relu(x)
        
        out1 = self.dec1(x)
        out2 = self.dec2(x)
        
        return out1, out2
    
    
    
class FeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu= nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        
        x = self.maxpool(x)
        return x
        

class FeatureDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.t_conv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.t_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.t_conv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.t_conv4 = nn.ConvTranspose2d(16, 3, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.relu= nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.t_conv1(x)
        x = self.relu(x)
        x = self.t_conv2(x)
        x = self.relu(x)
        x = self.t_conv3(x)
        x = self.relu(x)
        x = self.t_conv4(x)
        
        x = self.sigmoid(x)
        
        return x
        
    
    
    
    
    
    