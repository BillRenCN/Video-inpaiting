import os
import shutil
from argparse import ArgumentParser
import re

def process_files(input_folder, output_folder):
    # Ensure output folders exist
    masks_folder = os.path.join(output_folder, 'masks')
    os.makedirs(masks_folder, exist_ok=True)

    # Get list of all image files from the input folder
    images = [f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')]
    images.sort()  # Sort to maintain consistency

    # Iterate over all images and rename them
    for idx, filename in enumerate(images):
        # Create new filenames for original and mask
        new_image_name = f"{idx:06d}.jpg"
        mask_pattern = re.compile(rf"1\.0_{filename.split('.')[0]}__.*mask\.png")

        # Paths for the original image and mask
        original_image_path = os.path.join(input_folder, filename)
        mask_image_path = None

        # Find the corresponding mask file
        for mask_filename in os.listdir(output_folder):
            if mask_pattern.match(mask_filename):
                mask_image_path = os.path.join(output_folder, mask_filename)
                break

        # Paths for the renamed images in the output folders
        new_original_image_path = os.path.join(input_folder, new_image_name)
        new_mask_image_path = os.path.join(masks_folder, new_image_name)

        # Rename original image in place
        os.rename(original_image_path, new_original_image_path)

        # Copy and rename mask image if it exists
        if mask_image_path and os.path.exists(mask_image_path):
            shutil.copy2(mask_image_path, new_mask_image_path)
        else:
            print(f"Warning: Corresponding mask not found for {filename}")

    print("Processing completed.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input_folder', help='Path to the original images folder', required=True)
    parser.add_argument('--output_folder', help='Path to the output folder produced by the models', required=True)
    args = parser.parse_args()

    process_files(args.input_folder, args.output_folder)