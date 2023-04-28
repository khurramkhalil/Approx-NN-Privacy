import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cdist
from skimage.metrics import structural_similarity as compare_ssim

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


# Aggressive quantization
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model = torch.quantization.fuse_modules(model, [['conv1', 'relu1'], ['conv2', 'relu2']])
model_prepared = torch.quantization.prepare(model)
model_calibrated = model_prepared  # Replace this line with calibration using representative dataset if needed
model_quantized = torch.quantization.convert(model_calibrated, inplace=True)

def model_inversion_attack(model, device, target_class, steps, lr):
    model.eval()
    input_data = torch.randn(1, 1, 28, 28, device=device, requires_grad=True)
    optimizer = optim.SGD([input_data], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()

        with torch.no_grad():
            output = model(input_data)
            output.requires_grad = True

        loss = -output[0, target_class]  # Maximize the target class probability
        loss.backward()
        optimizer.step()

    return input_data.detach().cpu().numpy()[0, 0]

def evaluate_reconstructed_image(reconstructed_image, model, test_dataset, target_class):
    # Nearest neighbor search
    test_data, test_labels = zip(*[(img.numpy(), label) for img, label in test_dataset])
    test_data = np.array(test_data).reshape(len(test_data), -1)
    target_class_indices = [i for i, label in enumerate(test_labels) if label == target_class]

    distances = cdist(reconstructed_image.reshape(1, -1), test_data[target_class_indices], metric='euclidean')
    nearest_neighbor_index = np.argmin(distances)
    nearest_neighbor_label = test_labels[target_class_indices[nearest_neighbor_index]]

    # print(f"Nearest neighbor label: {nearest_neighbor_label}")
    # print(f"Nearest neighbor distance: {distances[0, nearest_neighbor_index]}")

    # Classifier confidence
    reconstructed_image_tensor = torch.tensor(reconstructed_image.reshape(1, 1, 28, 28)).to(device)
    with torch.no_grad():
        model.eval()
        output = model(reconstructed_image_tensor)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()

    confidence = probabilities[0, target_class]
    # print(f"Classifier confidence for target class: {confidence}")

    return nearest_neighbor_label, distances[0, nearest_neighbor_index], confidence

def evaluate_ssim(reconstructed_image, train_dataset, target_class, num_samples=10):
    target_samples = [img.numpy()[0] for img, label in train_dataset if label == target_class][:num_samples]
    ssim_values = [compare_ssim(reconstructed_image, sample, data_range=sample.max() - sample.min()) for sample in target_samples]
    mean_ssim = np.mean(ssim_values)
    return mean_ssim


model_hist = {}
new_model_hist = {}
for i in range(10):
    target_class = i
    steps = 1000
    learning_rate = 0.1

    reconstructed_image = model_inversion_attack(model, device, target_class, steps, learning_rate)
    new_reconstructed_image = model_inversion_attack(model_quantized, device, target_class, steps, learning_rate)

    nearest_neighbor_label, nearest_neighbor_distance, classifier_confidence = evaluate_reconstructed_image(
        reconstructed_image, model, test_dataset, target_class)
    new_nearest_neighbor_label, new_nearest_neighbor_distance, new_classifier_confidence = evaluate_reconstructed_image(
        new_reconstructed_image, model_quantized, test_dataset, target_class)
    
    mean_ssim = evaluate_ssim(reconstructed_image, test_dataset, target_class)
    new_mean_ssim = evaluate_ssim(new_reconstructed_image, test_dataset, target_class)

    model_hist[i] = [nearest_neighbor_label, nearest_neighbor_distance, classifier_confidence, mean_ssim]
    new_model_hist[i] = [new_nearest_neighbor_label, new_nearest_neighbor_distance, new_classifier_confidence, new_mean_ssim]


    # model_hist[i] = [mean_ssim]
    # new_model_hist[i] = [new_mean_ssim]    


# plt.imshow(reconstructed_image, cmap='gray')
# plt.title(f"Reconstructed Image (Target Class: {target_class})")

# Higher True count supports the idea [Use '>' operator to see which is higher]
comparison = [[i[-1], j[-1]] for i, j in zip(model_hist.values(), new_model_hist.values())]
print(comparison)
distance = [[i[1], j[1]] for i, j in zip(model_hist.values(), new_model_hist.values())]
print(distance)
# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 10))
fig.suptitle('Model Inversion Attack on Complete and Approx Model')
highlight_difference = False

x = np.arange(10)
y1 = np.array([i[0] for i in comparison])
y2 = np.array([i[1] for i in comparison])
ax1.scatter(x, y1, label='SSIM of Complete Model')
ax1.scatter(x, y2, label='SSIM of Approx Model')
if highlight_difference:
    # Shade the area between the two lines
    ax1.fill_between(x, y1, y2, where=y1>=y2, interpolate=True, alpha=0.5, color='green')
    ax1.fill_between(x, y1, y2, where=y1<y2, interpolate=True, alpha=0.5, color='red')

else:
    colors = ['red'if i>j else 'green' for i,j in comparison]
    ax1.vlines(x, y1, y2, linestyle='dashed', colors=colors, label='Red support hypothesis')
ax1.grid(True)
ax1.set_xticks(x)
ax1.set_xlabel('Classes')
ax1.set_ylabel('SSIM')
ax1.set_title('Structural Similarity Index')
ax1.legend()

z1 = np.array([i[0] for i in distance])
z2 = np.array([i[1] for i in distance])
ax2.scatter(x, z1, label='NN Distance of Complete Model')
ax2.scatter(x, z2, label='NN Distance of Approx Model')
if highlight_difference:
    # Shade the area between the two lines
    ax2.fill_between(x, z1, z2, where=z1>=z2, interpolate=True, alpha=0.5, color='green')
    ax2.fill_between(x, z1, z2, where=z1<z2, interpolate=True, alpha=0.5, color='red')
else:
    colors = ['red'if i>j else 'green' for i,j in distance]
    ax2.vlines(x, z1, z2, linestyle='dashed', colors=colors, label='Red support hypothesis')
ax2.grid(True)
ax2.set_xticks(x)
ax2.set_xlabel('Classes')
ax2.set_ylabel('Nearest Neighbour Distance')
ax2.set_title('Quality of the reconstructed image')
ax2.legend()

plt.show()
