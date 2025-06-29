import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from PIL import Image
import os

# filepath: c:\Users\yahya\WORK\trocr-arabic\src\preprocess.py
def load_data(csv_file):
    base_path = '/kaggle/working/arabic_images'  # Base path for images
    data = pd.read_csv(csv_file)
    image_paths = data['image'].tolist()
    full_image_paths = [os.path.join(base_path, path) for path in image_paths]  
    for path in full_image_paths:
        if not os.path.exists(path):
            print(f"File not found: {path}")
    return full_image_paths, data['text'].tolist()

def preprocess_images(image_paths):
    images = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        images.append(img)
    return images

def tokenize_text(texts, tokenizer):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

def split_dataset(images, texts, test_size=0.2, val_size=0.1):
    train_images, temp_images, train_texts, temp_texts = train_test_split(images, texts, test_size=test_size)
    val_size_adjusted = val_size / (1 - test_size)
    val_images, test_images, val_texts, test_texts = train_test_split(temp_images, temp_texts, test_size=val_size_adjusted)
    return train_images, train_texts, val_images, val_texts, test_images, test_texts

def main(csv_file, tokenizer_name):
    images, texts = load_data(csv_file)
    images = preprocess_images(images)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenized_texts = tokenize_text(texts, tokenizer)
    train_images, train_texts, val_images, val_texts, test_images, test_texts = split_dataset(images, texts)

    return train_images, train_texts, val_images, val_texts, test_images, test_texts, tokenized_texts

if __name__ == "__main__":
    csv_file = '/kaggle/working/arabic_images/annotations.csv'
    tokenizer_name = 'aubmindlab/bert-base-arabertv02' 

    # Call the main function
    train_images, train_texts, val_images, val_texts, test_images, test_texts, tokenized_texts = main(csv_file, tokenizer_name)

    # Data verification
    print("=== Data Verification ===")
    print(f"Number of training images: {len(train_images)}")
    print(f"Number of training texts: {len(train_texts)}")
    print(f"Number of validation images: {len(val_images)}")
    print(f"Number of validation texts: {len(val_texts)}")
    print(f"Number of test images: {len(test_images)}")
    print(f"Number of test texts: {len(test_texts)}")

    # Image verification
    print("\n=== Sample Image Paths ===")
    print(f"First training image path: {train_images[0] if train_images else 'No training images found'}")
    print(f"First validation image path: {val_images[0] if val_images else 'No validation images found'}")
    print(f"First test image path: {test_images[0] if test_images else 'No test images found'}")

    # Text verification
    print("\n=== Sample Texts ===")
    print(f"First training text: {train_texts[0] if train_texts else 'No training texts found'}")
    print(f"First validation text: {val_texts[0] if val_texts else 'No validation texts found'}")
    print(f"First test text: {test_texts[0] if test_texts else 'No test texts found'}")

    # Tokenized text verification
    print("\n=== Tokenized Texts ===")
    print(f"Tokenized text sample: {tokenized_texts['input_ids'][0] if 'input_ids' in tokenized_texts else 'No tokenized texts found'}")
