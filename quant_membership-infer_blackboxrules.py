import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from art.attacks.inference.membership_inference import MembershipInferenceBlackBoxRuleBased
from art.estimators.classification import PyTorchClassifier

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
        self.fc1 = nn.Linear(9216, 10)

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dequant(x)
        return nn.functional.log_softmax(x, dim=1)

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Instantiate and train the model
device = torch.device('cpu')
model = Net().to(device)
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
if not load:
    for epoch in range(1, 5):  # 10 epochs
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    # Assuming you have already trained a model called 'model'
    model_path = "my_model.pt"
    torch.save(model.state_dict(), model_path)

else:
    # Load the state_dict into the model
    model_path = "my_model.pt"
    model.load_state_dict(torch.load(model_path))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Base Model Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

model.eval()
# Aggressive quantization
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model = torch.quantization.fuse_modules(model, [['conv1', 'relu1'], ['conv2', 'relu2']])
model_prepared = torch.quantization.prepare(model)
model_calibrated = model_prepared  # Replace this line with calibration using representative dataset if needed
model_quantized = torch.quantization.convert(model_calibrated, inplace=True)

models = [model, model_quantized]
model_train, model_test = [], []
quant_train, quant_test = [], []
classes = list(range(0, 10, 1))

for target_class in classes:

    for final_model in models:
        # Target class
        # target_class = 2

        # Test accuracy
        # test(final_model, device, test_loader)

        # Prepare ART classifier
        art_classifier = PyTorchClassifier(
            model=final_model.to('cpu'),
            clip_values=(0, 1),
            loss=criterion,
            optimizer=optimizer,
            input_shape=(1, 28, 28),
            nb_classes=10,
        )

        # Membership Inference Attack
        attack = MembershipInferenceBlackBoxRuleBased(art_classifier)
        train_indices = [i for i, (_, target) in enumerate(train_dataset) if target == target_class][:500]
        test_indices = [i for i, (_, target) in enumerate(test_dataset) if target == target_class][:500]
        train_samples = torch.stack([train_dataset[i][0] for i in train_indices], dim=0).numpy()
        test_samples = torch.stack([test_dataset[i][0] for i in test_indices], dim=0).numpy()

        # attack.fit(x=train_samples, y=np.eye(10)[np.repeat(target_class, len(train_indices))], test_x=test_samples, test_y=np.eye(10)[np.repeat(target_class, len(test_indices))])

        inferred_train = attack.infer(train_samples, np.eye(10)[np.repeat(target_class, len(train_indices))])
        train_acc = np.sum(inferred_train) / len(inferred_train)
        print(f"Correct membership predictions on Train set: {train_acc * 100}%")

        inferred_test = attack.infer(test_samples, np.eye(10)[np.repeat(target_class, len(test_indices))])
        test_acc = np.sum(inferred_test) / len(inferred_test)
        print(f"Correct membership predictions on Test set: {test_acc * 100}%\t")
        
        if final_model == model:
            model_train.append(train_acc)
            model_test.append(test_acc)
        else:
            quant_train.append(train_acc)
            quant_test.append(test_acc)
        
        # def calc_precision_recall(predicted, actual, positive_value=1):
        #     score = 0  # both predicted and actual are positive
        #     num_positive_predicted = 0  # predicted positive
        #     num_positive_actual = 0  # actual positive
        #     for i in range(len(predicted)):
        #         if predicted[i] == positive_value:
        #             num_positive_predicted += 1
        #         if actual[i] == positive_value:
        #             num_positive_actual += 1
        #         if predicted[i] == actual[i]:
        #             if predicted[i] == positive_value:
        #                 score += 1
            
        #     if num_positive_predicted == 0:
        #         precision = 1
        #     else:
        #         precision = score / num_positive_predicted  # the fraction of predicted “Yes” responses that are correct
        #     if num_positive_actual == 0:
        #         recall = 1
        #     else:
        #         recall = score / num_positive_actual  # the fraction of “Yes” responses that are predicted correctly

        #     return precision, recall

        # precision, recall = calc_precision_recall(inferred_test, np.ones(len(inferred_test)))
        # # print(f"The presision and recall of attack is: {precision * 100}%, {recall * 100}%")

print('abc', model_test)
print('abc', quant_test)