import torch
import cv2
import numpy as np
import torch.nn.functional as F
from src.model import DeepLabV3Plus  # or U-Net if using U-Net
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
import os

def load_model(checkpoint_path, device):
    model = DeepLabV3Plus(num_classes=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image_path):
    transform = Compose([
        Resize(256, 256),  # Ensure this matches your training input size
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image)
    return transformed['image'].unsqueeze(0)  # Add batch dimension

def postprocess_mask(mask, original_height, original_width):
    mask = F.interpolate(mask, size=(original_height, original_width), mode='bilinear', align_corners=False)
    mask = mask.squeeze().cpu().numpy()  # Remove batch and channel dimensions
    mask = (mask > 0.3).astype(np.uint8) * 255  # Thresholding at 0.5 and converting to binary
    # mask = (mask > 0.5).astype(np.uint8) * 255  # Thresholding at 0.5 and converting to binary
    return mask

def run_inference(image_path, model, device):
    original_image = cv2.imread(image_path)
    original_height, original_width = original_image.shape[:2]
    
    image = preprocess_image(image_path).to(device)
    with torch.no_grad():
        output = model(image)
        mask = postprocess_mask(output, original_height, original_width)
    return mask

# def run_inference(image_path, model, device):
#     image = preprocess_image(image_path).to(device)
#     with torch.no_grad():
#         output = model(image)
#         mask = postprocess_mask(output)
#     return mask

def save_mask(mask, output_path):
    cv2.imwrite(output_path, mask)

if __name__ == "__main__":
    # Update the paths to match the correct file naming
    # image_path = "data/my_dataset/img_dir/train/ADE_train_00000001_processed.png"
    image_path = "data/my_dataset/img_dir/val/ADE_val_00000002_processed.png"
    checkpoint_path = "checkpoints/deeplabv3plus_epoch90.pth"
    output_path = "output_mask.png"
    
    # Device setup (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model(checkpoint_path, device)
    
    # Run inference
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
    else:
        mask = run_inference(image_path, model, device)
        # Save the output mask
        save_mask(mask, output_path)
        print(f"Inference completed. Mask saved at {output_path}")
