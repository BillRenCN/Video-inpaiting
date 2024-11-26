import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from src.model import DeepLabV3Plus  # Assuming DeepLabV3Plus is being used
from src.dataset import get_dataloaders
import matplotlib.pyplot as plt
import os

def train_model(train_loader, val_loader, device, num_epochs=300, save_interval=30, model_save_path="checkpoints/deeplabv3plus_epoch{}.pth", lr=2e-4, weight_decay=1e-4, step_size=30, gamma=0.3, plot_path="training_plot.png"):
    model = DeepLabV3Plus(num_classes=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Lists to store loss and accuracy
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # for images, masks in val_loader:
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        scheduler.step()

        avg_val_loss = validate_model(model, val_loader, device, criterion)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(), model_save_path.format(epoch + 1))
            print(f'Model checkpoint saved at epoch {epoch+1}')
    
    # Plot and save the loss curves
    plot_and_save(train_losses, val_losses, plot_path)

    print('Training complete.')

def validate_model(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss

def plot_and_save(train_losses, val_losses, plot_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    print(f'Loss plot saved as {plot_path}')
