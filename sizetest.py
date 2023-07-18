# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
import os
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read data
data_dir = './data'
sets = ['test', 'train', 'valid']

composed_transform = transforms.Compose([
    transforms.Grayscale(),  # Convert image to grayscale
    transforms.Resize((128, 128)),  # Adjust the desired size
    transforms.ToTensor()  # Convert image to tensor
])

image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), transform=composed_transform)
    for x in sets
}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=6, shuffle=True, num_workers=0) for x in sets}

# Take a look at some train data
dataset_name = 'train'
data_loader = dataloaders[dataset_name]

# Randomly select a batch from the data loader
images, labels = next(iter(data_loader))

# Print out the shape
x = images[0]
print("Initial shape:", x.shape)

# Define class labels
class_labels = image_datasets[dataset_name].classes

# Display the initial image
plt.imshow(x.permute(1, 2, 0).squeeze(), cmap='gray')
plt.axis('off')
plt.show()

# Apply convolutions
conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
x = conv1(x)
print("Shape after conv1:", x.shape)

pool = nn.MaxPool2d(kernel_size=2, stride=2)
x = pool(x)
print("Shape after pool1:", x.shape)

conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
x = conv2(x)
print("Shape after conv2:", x.shape)

x = pool(x)
print("Shape after pool2:", x.shape)

conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
x = conv3(x)
print("Shape after conv3:", x.shape)

x = pool(x)
print("Shape after pool3:", x.shape)

conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
x = conv4(x)
print("Shape after conv4:", x.shape)

x = pool(x)
print("Shape after pool4:", x.shape)

conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
x = conv5(x)
print("Shape after conv5:", x.shape)

x = pool(x)
print("Shape after pool5:", x.shape)

# Flatten for fully connected layers
x = x.view(x.size(0), -1)
print("Shape after flattening:", x.shape)

# Apply linear layers
fc1 = nn.Linear(in_features=16, out_features=128)
x = fc1(x)
print("Shape after fc1:", x.shape)

fc2 = nn.Linear(in_features=128, out_features=64)
x = fc2(x)
print("Shape after fc2:", x.shape)

num_classes = len(class_labels)
fc3 = nn.Linear(in_features=64, out_features=num_classes)
x = fc3(x)
print("Shape after fc3:", x.shape)
