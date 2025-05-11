import os
import torch
import timm
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

data_dir = "spectrograms"
num_classes = 5
batch_size = 64
num_epochs = 32
learning_rate = 3e-5
save_model_path = "vit_emotion_model.pth"

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Load Dataset
dataset = ImageFolder(root=data_dir, transform=transform)
print("Class to index mapping:", dataset.class_to_idx)

targets = np.array([sample[1] for sample in dataset])

train_indices, val_indices = train_test_split(
    np.arange(len(targets)),
    test_size=0.2,
    stratify=targets,
    random_state=42
)

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class ViTEmotionClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super(ViTEmotionClassifier, self).__init__()
        self.model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
        nn.init.xavier_uniform_(self.model.head.weight)
        nn.init.zeros_(self.model.head.bias)

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTEmotionClassifier(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Evaluating model
def evaluate_model(model, val_loader, device):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    return acc

def compute_validation_loss(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
    return total_val_loss / len(val_loader)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    val_loss = compute_validation_loss(model, val_loader, criterion, device)
    val_acc = evaluate_model(model, val_loader, device)

    print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_acc:.4f}")

    scheduler.step()

print("Training complete.")

torch.save(model.state_dict(), save_model_path)
print(f"Model saved: {save_model_path} with {val_acc:.4f} accuracy.")

