
# Arabic OCR with TrOCR

This project trains a TrOCR model for optical character recognition (OCR) on Arabic text. It involves generating synthetic Arabic text images, preprocessing the data, and training a model using the Transformers library. Below are the detailed steps, testing instructions, and guidance for running or training the project.

## Project Steps

### 1. Data Generation
- **Dataset:** Utilized the "yeeaee/arabic-larg" dataset from Hugging Face to source Arabic text samples.
- **Image Generation:** Created synthetic images by rendering the text with various Arabic fonts (e.g., IBM Plex Sans Arabic Bold, Cairo Regular), background colors (e.g., white, light gray), and text colors (e.g., black, red, blue). The images are fixed at 384x384 pixels.
- **Text Fitting:** Adjusted font sizes dynamically to fit text within the image width (384 pixels minus margins) and wrapped long text into multiple lines.
- **Output:** Generated 50,000 images (3 images per text sample) and saved them in the `arabic_images` directory, along with their corresponding texts in `annotations.csv`.

### 2. Data Preprocessing
- **Input:** Loaded the `annotations.csv` file containing image paths and text labels.
- **Validation:** Checked for missing image files and printed warnings if any were not found in the specified directory (`/kaggle/working/arabic_images`).
- **Image Preprocessing:** Opened each image and converted it to RGB format using Pillow.
- **Text Tokenization:** Tokenized the text using the 'aubmindlab/bert-base-arabertv02' tokenizer with padding and truncation.
- **Dataset Splitting:** Split the dataset into training (70%), validation (10%), and test (20%) sets using scikit-learn's `train_test_split`.

### 3. Model Training
- **Configuration:** Loaded training parameters (e.g., learning rate, batch size, epochs) from a `config.yaml` file.
- **Custom Dataset:** Defined `ArabicTextDataset` to load images and texts from the CSV, applying transformations like resizing to a specified size (e.g., 384x384) and converting to tensors.
- **DataLoader:** Created a DataLoader for batching and shuffling the training data.
- **Model Setup:** Loaded the TrOCR processor and VisionEncoderDecoderModel from a pre-trained checkpoint specified in the config.
- **Training Loop:** Trained the model using the AdamW optimizer, with forward passes, loss calculation, backpropagation, and periodic checkpoint saving.
- **Device:** Utilized GPU (CUDA) if available, otherwise CPU.

### 4. Testing the Model
After training, you can test the model on new images to predict Arabic text. Below is an example script to perform inference:

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Load the trained processor and model
processor = TrOCRProcessor.from_pretrained('path/to/trained/model')
model = VisionEncoderDecoderModel.from_pretrained('path/to/trained/model')

# Load and preprocess the test image
image = Image.open('path/to/test/image.png').convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values

# Generate text prediction
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Predicted text:", generated_text)
```

- Replace `'path/to/trained/model'` with the directory where your trained model is saved (e.g., a checkpoint folder).
- Ensure the test image is in a compatible format (e.g., PNG, 384x384 pixels).

### 5. Running or Training the Project
To run or train the project from scratch, follow these instructions:

#### Prerequisites
- **Dependencies:** Install the required Python libraries:
  ```bash
  pip install pandas scikit-learn transformers pillow torch pyyaml datasets
  ```
- **Fonts:** Ensure Arabic fonts (e.g., listed in the data generation script) are available in the specified directory (default: `/kaggle/input/fonts11/New folder`).
- **Hardware:** A GPU is recommended for faster training, though CPU is supported.

#### Steps
1. **Generate Synthetic Data:**
   - Run the data generation script to create images and annotations:
     ```bash
     python generate_data.py
     ```
   - Adjust `output_dir`, `fonts_dir`, and `num_samples` in the script if needed.

2. **Preprocess the Data:**
   - Run the preprocessing script to prepare the dataset:
     ```bash
     python preprocess.py
     ```
   - Ensure `csv_file` points to `arabic_images/annotations.csv` and update `base_path` if necessary.

3. **Train the Model:**
   - Configure `config.yaml` with your settings (e.g., model name, learning rate, output directory).
   - Run the training script:
     ```bash
     python train.py
     ```
   - Checkpoints will be saved in the output directory specified in the config.

4. **Test the Model:**
   - Use the inference script provided above with your trained model and test images.

#### Notes
- Ensure all file paths (e.g., CSV, images, fonts) match your local setup.
- Modify the `base_path` in `preprocess.py` or `images_dir` in `train.py` if your directory structure differs.
- The project assumes a Kaggle-like environment (`/kaggle/working/`); adjust paths for local use.


## License

This project is licensed under the [MIT License](./LICENSE).
