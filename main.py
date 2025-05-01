import torch
from preprocess import generate_spectrograms
from dataset import get_dataloaders
from model import create_model
from train import train_model
from evaluate import evaluate_model

if __name__ == "__main__":
    data_dir = "data/"
    spectrogram_dir = "spectrograms/"
    
    # Preprocess and generate spectrograms
    generate_spectrograms(data_dir, spectrogram_dir)

    # Prepare DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(spectrogram_dir)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_labels = len(train_loader.dataset.dataset.label_mapping)
    model = create_model(num_labels)

    # Define optimizer and loss function
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)

    # Train the model
    train_model(model, train_loader, val_loader, num_epochs=10, criterion=criterion, optimizer=optimizer, device=device)

    # Evaluate the model
    evaluate_model(model, test_loader, device)
