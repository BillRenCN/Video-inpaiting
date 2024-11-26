import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader 
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SubtitleDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.img_names = sorted([f for f in os.listdir(img_dir) if f.endswith('_processed.png')])
        self.transform = transform
    
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Generate corresponding mask name
        mask_name = img_name.replace('_processed.png', '_mask.png')
        mask_path = os.path.join(self.ann_dir, mask_name)
        
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Convert mask to binary (0 for black, 1 for white)
        mask = (mask > 127).astype('float32')
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Expand the mask to have a channel dimension: [1, 256, 256]
        mask = torch.unsqueeze(mask, dim=0)
        
        return image, mask

def get_dataloaders(train_img_dir, train_ann_dir, val_img_dir, val_ann_dir, batch_size=16):
    transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    train_dataset = SubtitleDataset(train_img_dir, train_ann_dir, transform=transform)
    val_dataset = SubtitleDataset(val_img_dir, val_ann_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
