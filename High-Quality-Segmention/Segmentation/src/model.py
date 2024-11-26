import torch
import torch.nn as nn
import torchvision.models.segmentation as models

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=1, backbone='resnet50'):
        super(DeepLabV3Plus, self).__init__()
        # Load a pre-trained DeepLabV3 model with a specific backbone
        self.model = models.deeplabv3_resnet50(pretrained=True) if backbone == 'resnet50' else models.deeplabv3_resnet101(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))  # Adjust the final layer

    def forward(self, x):
        x = self.model(x)['out']
        return torch.sigmoid(x)

# Example instantiation:
# model = DeepLabV3Plus(num_classes=1, backbone='resnet50')
