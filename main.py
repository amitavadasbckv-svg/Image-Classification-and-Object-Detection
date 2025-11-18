from collections import Counter
import os


#classes = os.listdir(data_dir)
#count = {cls: len(os.listdir(os.path.join(data_dir, cls))) for cls in classes}

'''
def count_images_per_class(dataset_path):
    class_counts = {}
    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_dir):
            class_counts[class_name] = len(os.listdir(class_dir))
    return class_counts

train_dir = r"C:\\Users\\Contr\\Downloads\\classification_dataset\\train"
counts = count_images_per_class(train_dir)
total = sum(counts.values())
for cls, count in counts.items():
    print(f"{cls}: {count} images ({count/total*100:.2f}%)")

max_class = max(counts, key=counts.get)
min_class = min(counts, key=counts.get)

imbalance_ratio = counts[max_class] / counts[min_class]

print("Imbalance Ratio:", imbalance_ratio)

if imbalance_ratio > 1.5:
    print("⚠ Significant class imbalance detected!")
else:
    print("✔ No major class imbalance.")


import matplotlib.pyplot as plt

classes = list(counts.keys())
values = list(counts.values())

plt.bar(classes, values)
plt.xlabel("Classes")
plt.ylabel("Number of images")
plt.title("Class Distribution")
plt.xticks(rotation=45)
plt.show()

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_samples(dataset_dir, samples_per_class=3):
    classes = os.listdir(dataset_dir)
    classes = [c for c in classes if os.path.isdir(os.path.join(dataset_dir, c))]

    plt.figure(figsize=(12, 8))

    for i, cls in enumerate(classes):
        class_dir = os.path.join(dataset_dir, cls)
        images = os.listdir(class_dir)[:samples_per_class]

        for j, img_name in enumerate(images):
            img_path = os.path.join(class_dir, img_name)
            img = mpimg.imread(img_path)

            plt.subplot(len(classes), samples_per_class, i * samples_per_class + j + 1)
            plt.imshow(img)
            plt.title(cls)
            plt.axis("off")

    plt.tight_layout()
    plt.show()

train_dir = r"C:\\Users\\Contr\\Downloads\\classification_dataset\\train"
show_samples(train_dir, samples_per_class=3)
'''

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

train_data = datasets.ImageFolder(
    root=r"C:\\Users\\Contr\\Downloads\\classification_dataset\\train",
    transform=transform
)
val_data = datasets.ImageFolder(
    root=r"C:\\Users\\Contr\\Downloads\\classification_dataset\\valid",
    transform=transform
)
test_data = datasets.ImageFolder(
    root=r"C:\\Users\\Contr\\Downloads\\classification_dataset\\test",
    transform=transform
)
#print(train_data)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

train_images, train_labels = next(iter(train_loader))
val_images, val_labels = next(iter(val_loader))
test_images, test_labels = next(iter(test_loader))

'''

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5    # unnormalize if normalized
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

import torchvision

# Make a grid of the batch images
img_grid = torchvision.utils.make_grid(test_images[:8])  # show first 8 images

imshow(img_grid)
'''

import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn
import pandas as pd

# Example input image
x = torch.randn(1, 3, 224, 224)  # [batch, channels, height, width]

conv_layers = nn.Sequential(
    nn.Conv2d(3, 32, 3), nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3), nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 64, 3), nn.ReLU()
)

# Forward pass to find flattened size
with torch.no_grad():
    out = conv_layers(x)
    flattened_size = out.view(1, -1).shape[1]
#print("Flattened size:", flattened_size)

model = nn.Sequential(
    conv_layers,
    nn.Flatten(),
    nn.Linear(flattened_size, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

criterion = nn.CrossEntropyLoss()  # same as SparseCategoricalCrossentropy(from_logits=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Empty DataFrame to store metrics
history_df = pd.DataFrame(columns=['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])

num_epochs = 10

for epoch in range(num_epochs):
    # ===== Training =====
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss /= len(train_loader)
    train_acc = 100 * correct / total

    # ===== Validation =====
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100 * correct / total

    history_df = pd.concat([history_df, pd.DataFrame([{
        'epoch': epoch+1,
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'val_loss': val_loss,
        'val_accuracy': val_acc
    }])], ignore_index=True)


    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
import matplotlib.pyplot as plt

plt.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss')
plt.plot(history_df['epoch'], history_df['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history_df['epoch'], history_df['train_accuracy'], label='Train Accuracy')
plt.plot(history_df['epoch'], history_df['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# ==========================================
# 7. SAVE MODEL
# ==========================================

# (Option A) Save the entire model
torch.save(model, "model_full.pth")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(labels, preds)
prec = precision_score(labels, preds, average='binary')
rec = recall_score(labels, preds, average='binary')
f1 = f1_score(labels, preds, average='binary')

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(labels, preds, target_names=["bird", "drone"]))
cm = confusion_matrix(labels, preds)
print(cm)



