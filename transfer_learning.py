import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.amp import autocast, GradScaler
import pandas as pd


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
#print(train_data.classes)
num_classes = len(train_data.classes)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True,
                          num_workers=0)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

train_images, train_labels = next(iter(train_loader))
val_images, val_labels = next(iter(val_loader))
test_images, test_labels = next(iter(test_loader))

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=.0005)
# Empty DataFrame to store metrics
history_df = pd.DataFrame(columns=['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])


#scaler = GradScaler(device=device)   # ✅ new API (no deprecation warning)

train_len = len(train_data)
val_len = len(val_data)
epochs = 10

for epoch in range(epochs):
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


    print(f"Epoch {epoch+1}/{epochs} | "
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

# (Option A) Save the entire model
torch.save(model, "transfer_learning.pth")

