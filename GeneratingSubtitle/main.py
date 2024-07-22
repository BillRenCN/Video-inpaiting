import os
import random
from PIL import Image, ImageDraw, ImageFont

def read_words(filepath):
    with open(filepath, 'r') as file:
        return file.read().split()

def select_words_to_fit(draw, font, words, max_width):
    for num_words in range(6, 1, -1):  # Try 6 to 2 words
        selected_words = ' '.join(words[:num_words])
        text_bbox = draw.textbbox((0, 0), selected_words, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        if text_width <= max_width:
            return selected_words
    return ' '.join(words[:2])  # Fallback to 2 words if nothing fits

def get_random_words(words, min_words=2, max_words=6):
    start_index = random.randint(0, len(words) - max_words)
    num_words = random.randint(min_words, max_words)
    return words[start_index:start_index + num_words]

def process_image(image_path, output_image_path, mask_image_path, words, font_path='font.ttf', font_size=20):
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
        print(f"Warning: Could not load font at {font_path}. Using default font.")
    
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    image_width, image_height = image.size
    selected_words = select_words_to_fit(draw, font, words, image_width - 20)
    
    text_bbox = draw.textbbox((0, 0), selected_words, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    text_x = (image_width - text_width) // 2
    text_y = image_height - text_height - 10
    
    draw.text((text_x, text_y), selected_words, font=font, fill="white")
    image.save(output_image_path)
    
    mask_image = Image.new('RGB', (image_width, image_height), color='black')
    mask_draw = ImageDraw.Draw(mask_image)
    mask_draw.text((text_x, text_y), selected_words, font=font, fill="white")
    mask_image.save(mask_image_path)

def main():
    text_file_path = 'text.txt'
    images_folder = 'images'
    output_folder = 'output'
    mask_folder = 'masks'
    font_path = 'font.ttf'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)
    
    if not os.path.exists(font_path):
        print(f"Error: Font file not found at {font_path}. Please ensure the font file exists.")
        return
    
    words = read_words(text_file_path)
    image_files = [f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]
    
    for index, image_file in enumerate(image_files):
        image_path = os.path.join(images_folder, image_file)
        random_words = get_random_words(words)
        output_image_path = os.path.join(output_folder, f'processed_{index}.png')
        mask_image_path = os.path.join(mask_folder, f'mask_{index}.png')
        
        process_image(image_path, output_image_path, mask_image_path, random_words, font_path)

if __name__ == '__main__':
    main()
