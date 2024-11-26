import os
import shutil

def create_dataset_subset(base_dir, subset_type):
    """
    This function copies and renames files from the specified 'train' or 'val' subsets.
    """
    # Define the paths to the directories
    images_dir = os.path.join(base_dir, 'img_dir', subset_type)
    masks_dir = os.path.join(base_dir, 'ann_dir', subset_type)
    output_dir = os.path.join(base_dir, subset_type)  # Changed to generic 'subset_type'
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the list of image files and sort them to maintain matching pairs
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('_processed.png')])
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('_mask.png')])
    
    # Copy and rename files
    for image_file, mask_file in zip(image_files, mask_files):
        # Extract the numeric part from the file name
        file_num = image_file.split('_')[2]  # This splits the filename like 'ADE_val_00000001_processed.png' and picks '00000001'
        
        # Convert the numeric string into an integer to remove leading zeros, then back to string
        file_num = str(int(file_num))
        
        # Define the new file names
        new_image_name = f"{file_num}.jpg"
        new_mask_name = f"{file_num}.png"
        
        # Copy and rename the image and mask to the output directory
        shutil.copy(os.path.join(images_dir, image_file), os.path.join(output_dir, new_image_name))
        shutil.copy(os.path.join(masks_dir, mask_file), os.path.join(output_dir, new_mask_name))
        
    print(f"{subset_type.capitalize()} files copied and renamed in directory: {output_dir}")

# Usage
base_directory = 'my_dataset'
create_dataset_subset(base_directory, 'train')
# create_dataset_subset(base_directory, 'val')
