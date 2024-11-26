import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import progressbar
import cv2

from models.network.crm_transferCoord_transferFeat import CRMNet
from dataset import OfflineDataset_crm_pad32 as OfflineDataset
from dataset import SplitTransformDataset
from util.image_saver_crm import tensor_to_im, tensor_to_gray_im, tensor_to_seg
from util.hyper_para import HyperParameters
from eval_helper_crm import process_high_res_im, process_im_single_pass

import os
from os import path
from argparse import ArgumentParser
import time
import numpy as np
from Segmentation.src.model import DeepLabV3Plus  # Import for the initial segmentation step
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2

def make_coord(shape, ranges=None, flatten=True, device=None): #
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n, device=device).float() # , 
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

class Parser():
    def parse(self):
        self.default = HyperParameters()
        self.default.parse(unknown_arg_ok=True)

        parser = ArgumentParser()

        parser.add_argument('--dir', help='Directory with testing images')
        parser.add_argument('--model', help='Pretrained model')
        parser.add_argument('--output', help='Output directory')

        parser.add_argument('--global_only', help='Global step only', action='store_true')

        parser.add_argument('--L', help='Parameter L used in the paper', type=int, default=900)
        parser.add_argument('--stride', help='stride', type=int, default=450)

        parser.add_argument('--clear', help='Clear pytorch cache?', action='store_true')

        parser.add_argument('--ade', help='Test on ADE dataset?', action='store_true')

        args, _ = parser.parse_known_args()
        self.args = vars(args)

    def __getitem__(self, key):
        if key in self.args:
            return self.args[key]
        else:
            return self.default[key]

    def __str__(self):
        return str(self.args)

def load_deeplab_model(checkpoint_path, device):
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

# Main script
before_Parser_time = time.time()
print("\n before_Parser_time:", before_Parser_time)

# Parse command line arguments
para = Parser()
para.parse()
print('Hyperparameters: ', para)

# Device setup (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load DeepLabV3Plus model for initial coarse segmentation
# checkpoint_path = "./Segmentation/checkpoints/deeplabv3plus_epoch90.pth"
checkpoint_path = "./Segmentation/checkpoints/deeplabv3plus_epoch20_updated.pth"
deeplab_model = load_deeplab_model(checkpoint_path, device)

# Ensure output folder exists
os.makedirs(para['output'], exist_ok=True)

# First Inference Step: Generate Initial Masks
input_folder = para['dir']
output_folder = para['output']

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith('.png'):
        image_path = os.path.join(input_folder, filename)
        original_image = cv2.imread(image_path)
        original_height, original_width = original_image.shape[:2]
        
        # Copy the original image to the output folder with "__im.png" suffix
        im_output_path = os.path.join(output_folder, f"{filename.split('.')[0]}__im.png")
        cv2.imwrite(im_output_path, original_image)
        
        # Run inference with DeepLabV3Plus model
        image = preprocess_image(image_path).to(device)
        with torch.no_grad():
            output = deeplab_model(image)
            mask = postprocess_mask(output, original_height, original_width)
        
        # Prepare output filenames
        prefix = filename.split('.')[0]
        gt_output_path = os.path.join(output_folder, f"{prefix}__gt.png")
        seg_output_path = os.path.join(output_folder, f"{prefix}__seg.png")
        
        # Save the output masks
        cv2.imwrite(gt_output_path, mask)
        cv2.imwrite(seg_output_path, mask)
        
        print(f"Inference completed for {filename}. Masks saved at {gt_output_path} and {seg_output_path}")

# Construct CRMNet model for refining the segmentation
model = CRMNet(backend='resnet50').cuda()
model.load_state_dict(torch.load(para['model']))

batch_size = 1
memory_chunk = 50176 * 16

if para['ade']:
    val_dataset = SplitTransformDataset(output_folder, need_name=True, perturb=False, img_suffix='_im.png')
else:
    val_dataset = OfflineDataset(output_folder, need_name=True, resize=False, do_crop=False, padding=True)
val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=2)

model = model.eval()

epoch_start_time = time.time()
counting = 0
s_list = [0.125, 0.25, 0.5, 1.0]

# Second Inference Step: Refine the Coarse Masks with CRMNet
with torch.no_grad():
    for im, seg, gt, name, crm_data in progressbar.progressbar(val_loader):
        counting += 1
        im, seg, gt = im, seg, gt
        for k, v in crm_data.items():
            crm_data[k] = v.cuda()

        turns = len(s_list)
        for turn in range(turns):
            s = s_list[turn]
            images = {}
            im_ = F.interpolate(im, size=(round(im.shape[-2] * s), round(im.shape[-1] * s)), mode='bilinear', align_corners=True).cuda()
            seg_ = F.interpolate(seg, size=(round(im.shape[-2] * s), round(im.shape[-1] * s)), mode='bilinear', align_corners=True).cuda()

            transferFeat = None
            transferCoord = None
            for i in range(0, gt.shape[-2]*gt.shape[-1], memory_chunk):
                if transferFeat is None:
                    chunk_images, transferCoord, transferFeat = model(im_, seg_, coord=crm_data['coord'][:, i:i+memory_chunk, :], cell=crm_data['cell'][:, i:i+memory_chunk, :], transferCoord=transferCoord, transferFeat=transferFeat)
                else:
                    chunk_images = model(im_, seg_, coord=crm_data['coord'][:, i:i+memory_chunk, :], cell=crm_data['cell'][:, i:i+memory_chunk, :], transferCoord=transferCoord, transferFeat=transferFeat)
                if 'pred_224' not in images.keys():
                    images = chunk_images
                else:
                    for key in images.keys():
                        images[key] = torch.cat((images[key], chunk_images[key]), axis=1)

                if para['clear']:
                    torch.cuda.empty_cache()

            for key in images.keys(): 
                images[key] = images[key].view(images[key].shape[0], images[key].shape[1]//(gt.shape[-2]*gt.shape[-1]), *gt.shape[-2:])

            images['im'] = im
            images['seg_'+str(turn)] = seg
            images['gt'] = gt

            # Suppress close-to-zero segmentation input
            for b in range(seg.shape[0]):
                if (seg[b]+1).sum() < 2:
                    images['pred_224'][b] = 0

            # Save output images
            for i in range(im.shape[0]):
                if turn == 0:
                    cv2.imwrite(path.join(para['output'], '%s_im.png' % (name[i])),
                                cv2.cvtColor(tensor_to_im(im[i])[32:-32, 32:-32], cv2.COLOR_RGB2BGR))
                    cv2.imwrite(path.join(para['output'], '%s_seg.png' % (name[i])),
                                tensor_to_seg(images['seg_'+str(turn)][i])[32:-32, 32:-32])
                    cv2.imwrite(path.join(para['output'], '%s_gt.png' % (name[i])),
                                tensor_to_gray_im(gt[i])[32:-32, 32:-32])
                cv2.imwrite(path.join(para['output'], (str(s) + ('_%s_mask.png' % (name[i])))),
                            tensor_to_gray_im(images['pred_224'][i])[32:-32, 32:-32])
                cv2.imwrite(path.join(para['output'], (str(s) + ('_%s_01mask.png' % (name[i])))),
                            tensor_to_gray_im(images['pred_224'][i]>0.5)[32:-32, 32:-32])  # 0 1

print('Time taken: %.1f s' % (time.time() - epoch_start_time))