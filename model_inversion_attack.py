import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

load = True
seed = 42
# Make deterministic
np.random.seed(seed)
# Set the random seed for PyTorch
torch.manual_seed(seed)
# Configure deterministic settings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.fc2 = nn.Linear(9216, 10)

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        x = self.dequant(x)
        return nn.functional.log_softmax(x, dim=1)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
if not load:
    epochs = 10
    for epoch in range(epochs):
        train(model, device, train_loader, optimizer, criterion)
else:
    # Load the state_dict into the model
    model_path = "my_model.pt"
    model.load_state_dict(torch.load(model_path))

# To improve the efficacy of the attack, we can modify the loss function to include 
# an additional term that encourages the output image to be closer to a real image from 
# the same class. We can achieve this by using the target class's mean image as a reference.

# Compute the mean image for each class in the training dataset
mean_images = torch.zeros((10, 1, 28, 28), device=device)

for i in range(10):
    class_data = [img for img, label in train_dataset if label == i]
    class_data_tensor = torch.stack(class_data)
    mean_images[i] = torch.mean(class_data_tensor, axis=0)


# Update the model_inversion_attack function to include the mean image term:
def model_inversion_attack(model, device, target_class, steps, lr, mean_images, alpha=0.5):
    model.eval()
    input_data = torch.randn(1, 1, 28, 28, device=device, requires_grad=True)
    optimizer = optim.SGD([input_data], lr=lr)
    target_mean_image = mean_images[target_class]

    for _ in range(steps):
        optimizer.zero_grad()
        output = model(input_data)
        class_loss = -output[0, target_class]
        mean_image_loss = alpha * torch.mean((input_data - target_mean_image) ** 2)
        total_loss = class_loss + mean_image_loss
        total_loss.backward()
        optimizer.step()
    
    return input_data.detach().cpu().numpy()[0, 0]

target_class = 5
steps = 1000
learning_rate = 0.1
reconstructed_image = model_inversion_attack(model, device, target_class, steps, learning_rate, mean_images)

plt.imshow(reconstructed_image, cmap='gray')
plt.title(f"Reconstructed Image (Target Class: {target_class})")
plt.show()
