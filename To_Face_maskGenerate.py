import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import quad
import math
import numpy as np
import time
from scipy.interpolate import CubicSpline
import torch.nn.functional as F
from torch.autograd import Function
import pywt
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from torchvision.models import mobilenet_v2

class MobileNetDetection(nn.Module):
    def __init__(self, num_classes=1):
        super(MobileNetDetection, self).__init__()
        # Load a pre-trained MobileNetV2 model
        mobilenet = mobilenet_v2(pretrained=True)
        # Replace the classifier to suit our output size
        self.features = mobilenet.features
        self.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(mobilenet.last_channel, num_classes * 4)  # 4 coordinates for bounding box
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.classifier(x)
        return x
    

if __name__ == '__main__':
    model = MobileNetDetection()
    print(model)
    x = torch.randn(10, 1, 8, 8)
    y = model(x)
    print(y.shape)