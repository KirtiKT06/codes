import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import Network


transform = transforms.Compose([
    transforms.ToTensor(),              # (1, 28, 28), values in [0,1]
    transforms.Lambda(lambda x: x.view(-1))  # flatten to (784,)
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)
experiment_results = {}

model = Network()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.1)

train_losses = []
test_losses = []

num_epochs = 10
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        train_loss = criterion(outputs, labels)
        train_loss.backward()
        optimizer.step()
        running_loss += train_loss.item()
    avg_train_loss = running_loss/ len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {avg_train_loss:.4f}")

    model.eval()
    running_test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            test_loss = criterion(outputs, labels)
            running_test_loss += test_loss.item()

    avg_test_loss = running_test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    print(f"           Test loss:  {avg_test_loss:.4f}")
    model.train()  # switch back

    
model.eval()   # switch to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)              # logits
        predictions = outputs.argmax(dim=1) # predicted digit
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Accuracy: {accuracy:.4f}")

experiment_results["relu_30"] = {
    "train_loss": train_losses,
    "test_loss": test_losses,
    "accuracy": accuracy
}

import matplotlib.pyplot as plt

plt.plot(train_losses, label="Train loss")
plt.plot(test_losses, label="Test loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()