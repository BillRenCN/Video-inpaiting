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
    mask = (mask > 0.1).astype(np.uint8) * 255  # Thresholding at 0.5 and converting to binary
    return mask

def run_inference(image_path, model, device):
    original_image = cv2.imread(image_path)
    original_height, original_width = original_image.shape[:2]
    
    image = preprocess_image(image_path).to(device)
    with torch.no_grad():
        output = model(image)
        mask = postprocess_mask(output, original_height, original_width)
    return mask

def save_mask(mask, output_path):
    cv2.imwrite(output_path, mask)

if __name__ == "__main__":
    # Set up paths
    input_folder = "./frames"  # Folder containing input images
    checkpoint_path = "checkpoints/deeplabv3plus_epoch20_updated.pth"
    # checkpoint_path = "checkpoints/deeplabv3plus_epoch90.pth"
    output_folder = "output_masks"  # Folder to save output masks
    
    # Device setup (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_model(checkpoint_path, device)
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through all images in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            image_path = os.path.join(input_folder, filename)
            
            # Copy the original image to the output folder with "__im.png" suffix
            im_output_path = os.path.join(output_folder, f"{filename.split('.')[0]}__im.png")
            original_image = cv2.imread(image_path)
            cv2.imwrite(im_output_path, original_image)
            
            # Run inference
            mask = run_inference(image_path, model, device)
            
            # Prepare output filenames
            prefix = filename.split('.')[0]
            gt_output_path = os.path.join(output_folder, f"{prefix}__gt.png")
            seg_output_path = os.path.join(output_folder, f"{prefix}__seg.png")
            
            # Save the output masks
            save_mask(mask, gt_output_path)
            save_mask(mask, seg_output_path)
            
            print(f"Inference completed for {filename}. Masks saved at {gt_output_path} and {seg_output_path}")