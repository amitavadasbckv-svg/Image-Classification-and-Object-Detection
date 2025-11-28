import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

from torchvision import models, transforms, datasets

transform = transforms.Compose([
    # 👉 1. Random crop (zoom + crop)
    transforms.RandomResizedCrop(
        size=224,            # final size
        scale=(0.8, 1.2),    # zoom out (0.8) / zoom in (1.2)
        ratio=(0.75, 1.33)   # aspect ratio variation
    ),

    # 👉 2. Random horizontal flip
    transforms.RandomHorizontalFlip(p=0.5),

    # 👉 3. Random rotation
    transforms.RandomRotation(degrees=25),

    # 👉 4. Random brightness / contrast / saturation
    transforms.ColorJitter(
        brightness=0.2,    # +-20% brightness
        contrast=0.2,
        saturation=0.2
    ),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)   # if input = 224x224
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

cnn_model = SimpleCNN(num_classes=2).to(device)

from torchvision import models

resnet_model = models.resnet18(pretrained=True)

# change last layer for binary classification
num_features = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_features, 2)

resnet_model = resnet_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_cnn = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
optimizer_resnet = torch.optim.Adam(resnet_model.parameters(), lr=0.0001)

train_data = datasets.ImageFolder(
        root=r"classification_dataset\\train",
        transform=transform
    )
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

for epoch in range(5):
    cnn_model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = cnn_model(images)
        loss = criterion(outputs, labels)

        optimizer_cnn.zero_grad()
        loss.backward()
        optimizer_cnn.step()

for epoch in range(5):
    resnet_model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = resnet_model(images)
        loss = criterion(outputs, labels)

        optimizer_resnet.zero_grad()
        loss.backward()
        optimizer_resnet.step()

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def eval_model(model, dataloader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='binary'),
        "recall": recall_score(y_true, y_pred, average='binary'),
        "f1": f1_score(y_true, y_pred, average='binary'),
    }
    return metrics

test_data = datasets.ImageFolder(
        root=r"classification_dataset\\test",
        transform=transform
    )


test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

cnn_results = eval_model(cnn_model, test_loader)
resnet_results = eval_model(resnet_model, test_loader)

print("CNN Metrics:", cnn_results)
print("ResNet Metrics:", resnet_results)

print("\n===== MODEL COMPARISON REPORT =====\n")
print(f"{'Metric':<15}{'CNN':<15}{'ResNet':<15}")
print("-" * 40)
print(f"{'Accuracy':<15}{cnn_results['accuracy']:<15.4f}{resnet_results['accuracy']:<15.4f}")
print(f"{'Precision':<15}{cnn_results['precision']:<15.4f}{resnet_results['precision']:<15.4f}")
print(f"{'Recall':<15}{cnn_results['recall']:<15.4f}{resnet_results['recall']:<15.4f}")
print(f"{'F1 Score':<15}{cnn_results['f1']:<15.4f}{resnet_results['f1']:<15.4f}")

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_cm(model, dataloader, title):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_cm(cnn_model, test_loader, "CNN Confusion Matrix")
plot_cm(resnet_model, test_loader, "ResNet Confusion Matrix")
