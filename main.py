import loading  # Import your data loading script
import torch
import torch.nn as nn
from torch.nn import Module, Conv2d, Linear, ReLU, MaxPool2d, CrossEntropyLoss, Flatten, LogSoftmax
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms import ToTensor, Compose, Resize
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
import time

class CNN(Module):
    def __init__(self, numChannels, classes):
        super(CNN, self).__init__()
        self.conv1 = Conv2d(in_channels=numChannels, out_channels=20, kernel_size=5, stride=1, padding=0)
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=0)
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=2, stride=2)

        # Output size after all layers
        self.fc1 = Linear(in_features=177 * 317 * 50, out_features=500)
        self.relu3 = ReLU()
        self.fc2 = Linear(in_features=500, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        output = self.logSoftmax(x)
        return output




# Preprocessing: Convert data lists to tensors
train_images = torch.stack([torch.Tensor(data[0]).view(1, 720, 1280) for data in loading.train_data])
train_labels = torch.tensor([label for _, label in loading.train_data])
test_images = torch.stack([torch.Tensor(data[0]).view(1, 720, 1280) for data in loading.test_data])
test_labels = torch.tensor([label for _, label in loading.test_data])

# Create TensorDatasets
train_dataset = TensorDataset(train_images, train_labels)
test_dataset = TensorDataset(test_images, test_labels)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(numChannels=1, classes=2).to(device)

# Optimizer and loss function
optimizer = Adam(model.parameters(), lr=1e-3)
criterion = nn.NLLLoss()

# Training loop
num_epochs = int(input("Enter the number of epochs: "))
for epoch in range(num_epochs=3):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.argmax(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct / total} %')
