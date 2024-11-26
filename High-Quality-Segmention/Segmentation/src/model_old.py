import torch
import torch.nn as nn
import torchvision.models as models

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.encoder = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.encoder_layers = list(self.encoder.children())
        
        # Encoder layers
        self.base = nn.Sequential(*self.encoder_layers[:-2])
        
        # Decoder layers to upsample to the original size
        self.upsample1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upsample4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        
        # Final output layer
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.base(x)  # Feature extraction
        
        # Upsampling to the original size
        x = self.upsample1(x)  # Output: [B, 256, 16, 16]
        x = self.upsample2(x)  # Output: [B, 128, 32, 32]
        x = self.upsample3(x)  # Output: [B, 64, 64, 64]
        x = self.upsample4(x)  # Output: [B, 64, 128, 128]
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  # Output: [B, 64, 256, 256]
        
        # Final convolution to produce the segmentation map
        x = self.out_conv(x)  # Output: [B, 1, 256, 256]
        
        return torch.sigmoid(x)
