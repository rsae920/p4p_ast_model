import torch
from dataloader import get_dataloaders
from model import create_model

# Define Training Function
def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader):.4f}")

        # Validate Model
        validate_model(model, val_loader, criterion, device)

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(pixel_values=images).logits
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

# Main Training Entry Point
if __name__ == "__main__":
    # Prepare DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(spectrogram_dir="spectrograms", batch_size=32)

    # Initialize Model
    num_labels = len(train_loader.dataset.dataset.label_mapping)
    model = create_model(num_labels)

    # Optimizer and Loss Function
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train Model
    train_model(model, train_loader, val_loader, num_epochs=10, criterion=criterion, optimizer=optimizer, device=device)
