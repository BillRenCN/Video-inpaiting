import os
import random
from PIL import Image, ImageDraw, ImageFont

def read_text_file(filepath, language):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read().replace('\n', '')
        if language in ['chinese', 'japanese']:
            # Return a list of characters
            return list(content)
        else:
            # Split by whitespace for English and Korean
            return content.split()

def select_text_to_fit(draw, font, text_units, max_width, min_units=2, max_units=10, language='english'):
    # Adjust the limits for Korean to avoid subtitles being too long
    if language == 'korean':
        max_units = min(max_units, 6)  # Limit Korean subtitles to a maximum of 6 units
    
    for num_units in range(max_units, min_units - 1, -1):
        selected_units = text_units[:num_units]
        selected_text = ''.join(selected_units)
        text_bbox = draw.textbbox((0, 0), selected_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        if text_width <= max_width:
            return selected_text
    # Fallback to min_units
    selected_units = text_units[:min_units]
    selected_text = ''.join(selected_units)
    return selected_text

def get_random_text_units(text_units, min_units=2, max_units=10, language='english'):
    if language == 'korean':
        # Limit max_units for Korean to prevent overly long subtitles
        max_units = min(max_units, 6)

    max_possible_units = min(max_units, len(text_units))
    min_possible_units = min(min_units, max_possible_units)
    num_units = random.randint(min_possible_units, max_possible_units)
    start_index = random.randint(0, len(text_units) - num_units)
    return text_units[start_index:start_index + num_units]

def draw_text_with_shadow(draw, position, text, font, text_color, shadow_color):
    x, y = position
    # Shadow offsets for a more pronounced shadow effect
    offsets = [(-2, -2), (-2, 0), (-2, 2), (0, -2), (0, 2), (2, -2), (2, 0), (2, 2)]
    # Draw shadow
    for offset in offsets:
        draw.text((x + offset[0], y + offset[1]), text, font=font, fill=shadow_color)
    # Draw main text
    draw.text((x, y), text, font=font, fill=text_color)

def draw_text_with_transparent_box(image, position, text, font, text_color, box_color, box_opacity):
    x, y = position
    draw = ImageDraw.Draw(image)
    # Get size of the text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    # Padding around text
    padding_x = 10
    padding_y = 5
    # Calculate box dimensions
    box_width = text_width + 2 * padding_x
    box_height = text_height + 2 * padding_y
    box_x = x - padding_x
    box_y = y - padding_y
    # Create a transparent box
    box = Image.new('RGBA', (int(box_width), int(box_height)), box_color + (int(box_opacity * 255),))
    # Paste the box onto the image
    image.paste(box, (int(box_x), int(box_y)), box)
    # Draw text over the box
    draw.text((x, y), text, font=font, fill=text_color)

def process_image(image_path, output_image_path, mask_image_path, text_units, font_path='font.ttf', font_size=20, language='english'):
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
        print(f"Warning: Could not load font at {font_path}. Using default font.")

    image = Image.open(image_path)
    image = image.convert('RGBA')  # Convert to RGBA for transparency handling
    draw = ImageDraw.Draw(image)

    image_width, image_height = image.size

    # Randomly select text units
    selected_text_units = get_random_text_units(text_units, language=language)
    # Select text that fits
    selected_text = select_text_to_fit(draw, font, selected_text_units, image_width - 40, language=language)

    # Recalculate text dimensions
    text_bbox = draw.textbbox((0, 0), selected_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Adjusted to consider padding when centering
    padding_x = 10
    padding_y = 5

    # Calculate positions
    text_x = (image_width - text_width) // 2
    text_y = image_height - text_height - 20  # Adjusted bottom margin

    # Randomly choose effect
    effect = random.choice(['shadow', 'transparent_box', 'none'])

    if effect == 'shadow':
        draw_text_with_shadow(draw, (text_x, text_y), selected_text, font, text_color="white", shadow_color="black")
    elif effect == 'transparent_box':
        draw_text_with_transparent_box(image, (text_x, text_y), selected_text, font, text_color="white", box_color=(0, 0, 0), box_opacity=0.5)
    else:  # 'none'
        draw.text((text_x, text_y), selected_text, font=font, fill="white")

    # Save the image (convert back to RGB if needed)
    output_image = image.convert('RGB')
    output_image.save(output_image_path)

    # Create mask image
    mask_image = Image.new('L', (image_width, image_height), color=0)
    mask_draw = ImageDraw.Draw(mask_image)
    # Draw the text in white (255)
    mask_draw.text((text_x, text_y), selected_text, font=font, fill=255)
    mask_image.save(mask_image_path)

def main():
    images_folder = 'images/training'
    generated_dataset_folder = 'generated_dataset'

    languages = ['chinese', 'english', 'japanese', 'korean']
    text_files = {
        'chinese': 'text_ch.txt',
        'english': 'text_en.txt',
        'japanese': 'text_ja.txt',
        'korean': 'text_ko.txt'
    }
    font_files = {
        'chinese': 'font_cn.ttf',
        'english': 'font_en.ttf',
        'japanese': 'font_ja.ttf',
        'korean': 'font_ko.ttf'
    }

    if not os.path.exists(generated_dataset_folder):
        os.makedirs(generated_dataset_folder)

    # Read text for each language
    texts = {}
    for language in languages:
        text_file_path = text_files[language]
        if not os.path.exists(text_file_path):
            print(f"Error: Text file not found for {language} at {text_file_path}. Please ensure the file exists.")
            return
        texts[language] = read_text_file(text_file_path, language)

    # Check font files
    for language in languages:
        font_path = font_files[language]
        if not os.path.exists(font_path):
            print(f"Error: Font file not found for {language} at {font_path}. Please ensure the font file exists.")
            return

    image_files = [f for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]

    counter = 1  # Initialize image numbering

    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        for language in languages:
            text_units = texts[language]
            font_path = font_files[language]
            output_image_path = os.path.join(generated_dataset_folder, f'{counter}.jpg')
            mask_image_path = os.path.join(generated_dataset_folder, f'{counter}.png')

            process_image(image_path, output_image_path, mask_image_path, text_units, font_path, language=language)

            counter += 1

if __name__ == '__main__':
    main()
