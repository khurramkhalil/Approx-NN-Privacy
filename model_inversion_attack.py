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


# To obtain better results, we can modify the attack to take advantage of multiple target class samples 
# in the training dataset. This will help guide the optimization process to produce a more convincing reconstructed image

# Update the model_inversion_attack function to include multiple target class samples
def model_inversion_attack(model, device, target_class, steps, lr, train_dataset, num_samples=10, alpha=0.5):
    model.eval()
    input_data = torch.randn(1, 1, 28, 28, device=device, requires_grad=True)
    optimizer = optim.SGD([input_data], lr=lr)

    target_samples = [img for img, label in train_dataset if label == target_class]
    target_samples = target_samples[:num_samples]
    target_samples = torch.stack(target_samples).to(device)

    for _ in range(steps):
        optimizer.zero_grad()
        output = model(input_data)

        class_loss = -output[0, target_class]
        sample_losses = [alpha * torch.mean((input_data - target_sample) ** 2) for target_sample in target_samples]
        mean_sample_loss = torch.mean(torch.stack(sample_losses))
        total_loss = class_loss + mean_sample_loss

        total_loss.backward()
        optimizer.step()
    
    return input_data.detach().cpu().numpy()[0, 0]



target_class = 5
steps = 1000
learning_rate = 0.1


reconstructed_image = model_inversion_attack(model, device, target_class, steps, learning_rate, train_dataset)

plt.imshow(reconstructed_image, cmap='gray')
plt.title(f"Reconstructed Image (Target Class: {target_class})")
plt.show()

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist

# Nearest neighbor search
train_data, train_labels = zip(*[(img.numpy(), label) for img, label in train_dataset])
train_data = np.array(train_data).reshape(len(train_data), -1)
target_class_indices = [i for i, label in enumerate(train_labels) if label == target_class]

distances = cdist(reconstructed_image.reshape(1, -1), train_data[target_class_indices], metric='euclidean')
nearest_neighbor_index = np.argmin(distances)
nearest_neighbor_label = train_labels[target_class_indices[nearest_neighbor_index]]

print(f"Nearest neighbor label: {nearest_neighbor_label}")
print(f"Nearest neighbor distance: {distances[0, nearest_neighbor_index]}")

# Classifier confidence
reconstructed_image_tensor = torch.tensor(reconstructed_image.reshape(1, 1, 28, 28)).to(device)
with torch.no_grad():
    model.eval()
    output = model(reconstructed_image_tensor)
    probabilities = torch.softmax(output, dim=1).cpu().numpy()

confidence = probabilities[0, target_class]
print(f"Classifier confidence for target class: {confidence}")
