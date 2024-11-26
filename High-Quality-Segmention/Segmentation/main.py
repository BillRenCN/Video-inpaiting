from src.train import train_model
from src.dataset import get_dataloaders
from src.utils import set_seed
import torch

# Set seed for reproducibility
set_seed()

# Directories for training and validation data
train_img_dir = "/home/bill/Documents/Research/Video-inpaiting/High-Quality-Segmention/my_dataset/train"
val_img_dir = "/home/bill/Documents/Research/Video-inpaiting/High-Quality-Segmention/my_dataset/validation"

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Get data loaders
train_loader, val_loader = get_dataloaders(train_img_dir, val_img_dir, batch_size=64)

# Train the model
# train_model(train_loader, val_loader, device, num_epochs=50, save_interval=10, model_save_path="checkpoints/deeplabv3plus_epoch{}_updated.pth", lr=2e-4, weight_decay=3e-5, step_size=30, gamma=0.3, plot_path="training_plot.png")
train_model(train_loader, val_loader, device, num_epochs=50, save_interval=10, model_save_path="checkpoints/deeplabv3plus_epoch{}_updated.pth", lr=2e-4, weight_decay=3e-5, step_size=30, gamma=0.3, plot_path="training_plot.png")
# train_model(train_loader, val_loader, device, num_epochs=150, save_interval=30, model_save_path="checkpoints/deeplabv3plus_epoch{}_new.pth", lr=1e-3, weight_decay=1e-5, step_size=30, gamma=0.3, plot_path="training_plot.png")
