import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import csv
from datasets import load_dataset
import textwrap

 # Arabic data set from hugging_face
dataset = load_dataset("yeeaee/arabic-larg")

# The file containing the fonts
fonts_dir = "/kaggle/input/fonts11/New folder"
arabic_fonts = [
    "arfonts-ibm-plex-sans-arabic-bold.ttf", "arfonts-cairo-regular.ttf", "arfonts-noto-naskh-arabic-regular.ttf", 
    "arfonts-amiri-quran.ttf", "arfonts-noto-naskh-arabic-bold.ttf", "arfonts-scheherazade-bold.ttf", 
    "arfonts-aref-ruqaa-bold.ttf", "arfonts-tajawal-black.ttf"
]
arabic_fonts = [os.path.join(fonts_dir, font) for font in arabic_fonts if os.path.exists(os.path.join(fonts_dir, font))]


background_colors = [(255, 255, 255), (240, 240, 240), (230, 230, 230), (250, 250, 200), (200, 220, 255)]
text_colors = [(0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 128, 0), (100, 50, 0)]

# The optimal size for training the model is 384
FIXED_WIDTH = 384
FIXED_HEIGHT = 384
MARGIN = 40  


num_samples = 50000  
images_per_text = 3  
img_count = 0
output_dir = "arabic_images"
os.makedirs(output_dir, exist_ok=True)


annotations = []

def fit_text_to_width(text, font_path, max_width):
    """Determine the Appropriate Font Size for the Text to Fit Within the Specified Width"""
    font_size = 60  
    min_font_size = 24  
    
    # Reduce Font Size Until the Text Fits Within the Available Width
    while font_size >= min_font_size:
        font = ImageFont.truetype(font_path, font_size)
        text_width = font.getlength(text)
        
        if text_width <= max_width:
            return font, text_width, font_size
            
        font_size -= 2
    
    # If the Minimum Size is Reached and the Text Doesn't Fit, Split It
    
    font = ImageFont.truetype(font_path, min_font_size)
    return font, font.getlength(text), min_font_size

def wrap_text(text, font, max_width):
    """Split Text into Multiple Lines if Too Long"""
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        
        test_line = ' '.join(current_line + [word]) if current_line else word
        test_width = font.getlength(test_line)
        
        if test_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                
                lines.append(word)
    
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

for i, sample in enumerate(dataset["train"]):
    if img_count >= num_samples:
        break
    
    text = sample.get("text", "").strip()
    if not text:
        continue

    for j in range(images_per_text):
        try:
            # choose random font
            
            font_path = random.choice(arabic_fonts)
            
            
            available_width = FIXED_WIDTH - (MARGIN * 2)
            
            
            font, text_width, font_size = fit_text_to_width(text, font_path, available_width)
            
            
            lines = wrap_text(text, font, available_width)
            
            
            background_color = random.choice(background_colors)
            text_color = random.choice(text_colors)
            img = Image.new("RGB", (FIXED_WIDTH, FIXED_HEIGHT), background_color)
            draw = ImageDraw.Draw(img)
            
            
            line_height = int(font_size * 1.3) 
            total_text_height = line_height * len(lines)
            
            
            y_start = (FIXED_HEIGHT - total_text_height) // 2
            
            # Draw Each Line of Text
            
            for idx, line in enumerate(lines):
                line_width = font.getlength(line)
                x_pos = (FIXED_WIDTH - line_width) // 2
                y_pos = y_start + (idx * line_height)
                draw.text((x_pos, y_pos), line, fill=text_color, font=font)

            # save photo
            
            img_path = os.path.join(output_dir, f"text_{img_count}.png")
            img.save(img_path)
            
            # Add comments 
            
            annotations.append({"image": img_path, "text": text})
            img_count += 1

        except Exception as e:
            print(f"❌  {img_count}: {e}")
            import traceback
            traceback.print_exc()

# Saving Comments to a CSV File

annotations_file = os.path.join(output_dir, "annotations.csv")
with open(annotations_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "text"])
    writer.writeheader()
    writer.writerows(annotations)
