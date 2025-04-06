import os
import yaml
import pandas as pd
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Clear any cached GPU memory
torch.cuda.empty_cache()

# Custom dataset class for Arabic OCR training
class ArabicTextDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Initialize dataset with image paths and corresponding text labels.
        
        Args:
            csv_file (str): Path to the CSV file containing 'image' and 'text' columns.
            transform (callable, optional): Transform to apply to each image.
        """
        self.base_path = os.path.dirname(os.path.abspath(csv_file))
        self.images_dir = os.path.join(self.base_path, "images")
        self.data = pd.read_csv(csv_file)
        self.image_paths = self.data['image'].values
        self.texts = self.data['text'].values
        self.transform = transform

        # Check for missing images in the dataset
        missing_images = []
        for img_path in self.image_paths:
            full_path = os.path.join(self.images_dir, img_path)
            if not os.path.exists(full_path):
                missing_images.append(img_path)
        
        if missing_images:
            print(f"Warning: {len(missing_images)} images not found!")
            print(f"First few missing images: {missing_images[:5]}")
            print(f"Looking for images in: {self.images_dir}")

    def __len__(self):
        """Return total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get one sample from the dataset.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: (image_tensor, target_text)
        """
        image_path = os.path.join(self.images_dir, self.image_paths[idx])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            raise
        text = self.texts[idx]
        if self.transform:
            image = self.transform(image)
        return image, text

# Function to train the OCR model
def train_model(model, processor, dataloader, optimizer, device, config):
    """
    Train the model on the given dataset.
    
    Args:
        model: VisionEncoderDecoderModel to train.
        processor: TrOCRProcessor to preprocess text.
        dataloader: DataLoader for training data.
        optimizer: Optimizer for training.
        device: Device to train on (CPU or GPU).
        config: Dictionary containing training configurations.
    """
    model.train()
    step = 0
    total_loss = 0
    
    print(f"Training on device: {device}")
    print(f"Number of epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    
    try:
        for epoch in range(config['num_epochs']):
            epoch_loss = 0
            for batch_idx, (images, texts) in enumerate(dataloader):
                images = images.to(device)

                # Tokenize the ground truth text
                inputs = processor(text=texts, return_tensors="pt", padding=True, 
                                truncation=True, max_length=config['max_seq_length']).to(device)

                # Forward pass
                outputs = model(images, labels=inputs['input_ids'])
                loss = outputs.loss

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step += 1
                epoch_loss += loss.item()
                
                # Log progress
                if batch_idx % 10 == 0:
                    print(f"Epoch [{epoch+1}/{config['num_epochs']}], "
                          f"Batch [{batch_idx}/{len(dataloader)}], "
                          f"Loss: {loss.item():.4f}")

                # Save checkpoint every X steps
                if step % config['save_steps'] == 0:
                    checkpoint_path = os.path.join(config['output_dir'], f"checkpoint-{step}")
                    model.save_pretrained(checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        # Save emergency checkpoint if something goes wrong
        model.save_pretrained(os.path.join(config['output_dir'], "emergency-checkpoint"))
        raise e

# Main function to run the full training pipeline
def main():
    """
    Load configuration, prepare dataset, and start training the model.
    """
    # Load configuration from YAML file
    with open(os.path.join(os.path.dirname(__file__), "../config.yaml"), "r") as file:
        config = yaml.safe_load(file)
    
    # Convert config values to appropriate types
    config['learning_rate'] = float(config['learning_rate'])
    config['batch_size'] = int(config['batch_size'])
    config['num_epochs'] = int(config['num_epochs'])
    config['max_seq_length'] = int(config['max_seq_length'])
    config['image_size'] = int(config['image_size'])
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)

    # Load dataset
    csv_file = os.path.join(os.path.dirname(__file__), "../data/data.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found at {csv_file}")
        
    print(f"Loading dataset from {csv_file}")
    dataset = ArabicTextDataset(
        csv_file, 
        transform=transforms.Compose([
            transforms.Resize((config['image_size'], config['image_size'])),
            transforms.ToTensor()
        ])
    )
    print(f"Dataset loaded with {len(dataset)} samples")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0  # Set to 0 for debugging
    )

    # Load processor and model
    processor = TrOCRProcessor.from_pretrained(config['model_name'])  
    model = VisionEncoderDecoderModel.from_pretrained(config['model_name'])
    
    # Configure model for training
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # Start training
    train_model(model, processor, dataloader, optimizer, device, config)

if __name__ == "__main__":
    main()
