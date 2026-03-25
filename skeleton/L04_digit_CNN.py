import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')  # Force CPU usage
if device.type == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

NUM_EPOCHS = 10

# Define CNN model with one convolutional and one pooling layer
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 1 input channels gray, 32 output channels, 3 kernel size
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2,2)

        # connected layer conpressed to neurons
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)
        # TODO: Make your convolutional, pooling, flattening, and fully connected layers here

    def forward(self, x):
        # math funciton applied to convul image
        # pooling to reduce map
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # "reform" shape, all we know is second dim is 1440
        # bridges convolutions to fully connect
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # softmax to get probability distribution
        return F.log_softmax(x, dim=1)

# Data transformation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load dataset
train_dataset = datasets.MNIST(root='', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='', train=False, transform=transform, download=True)
# Create batches
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Initialize model, loss, and optimizer
model = CNN()
model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader: # load one batch at a time
        images, labels = images.to(device), labels.to(device) # copying data to device (CPU/GPU)
        images = images.view(-1, 1, 28, 28) # reshape the image tensor
        optimizer.zero_grad() # resetting gradients
        output = model(images) # forward pass, automatically calling forward(x)
        loss = criterion(output, labels)
        loss.backward() # backpropagation
        optimizer.step() # weight updates
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

# Evaluate accuracy
correct, total = 0, 0
model.eval()
with torch.no_grad(): # disabling gradient tracking
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device) # copying data to device (CPU/GPU)
        
        images = images.view(-1, 1, 28, 28)
        output = model(images) # predict output class of image input
        # max(output,1) extract the class prediction from the output
        # by finding the class that has the highest score
        _, predicted = torch.max(output, 1) # 1 specifies the class dimension (column)
        total += labels.size(0) # number of samples in the current batch
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {correct / total * 100:.2f}%")

# Save the model parameters that can be loaded later
torch.save(model.state_dict(), "my_cnn.ph")
