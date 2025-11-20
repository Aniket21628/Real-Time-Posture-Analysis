"""
Training script for custom CNN posture detection model.

This script demonstrates how to train the custom CNN model on posture detection data.
Due to hardware limitations (no GPU available), this was designed but trained on
a different machine with GPU support.

Usage:
    python train_cnn.py --data_dir ./data/training --epochs 50 --batch_size 32
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path

from app_models.cnn_model import PostureCNN, PostureCNNDeep, train_model


class PostureDataset(Dataset):
    """
    Custom Dataset for loading posture images.
    
    Expected directory structure:
    data_dir/
        sitting_good/
            image1.jpg
            image2.jpg
            ...
        sitting_bad/
            image1.jpg
            image2.jpg
            ...
    """
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Root directory containing class subdirectories
            transform: Optional transform to be applied on images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = ['sitting_good', 'sitting_bad']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Load all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"Warning: Directory {class_dir} does not exist")
                continue
            
            for img_path in class_dir.glob('*.jpg'):
                self.samples.append((img_path, self.class_to_idx[class_name]))
            for img_path in class_dir.glob('*.png'):
                self.samples.append((img_path, self.class_to_idx[class_name]))
        
        print(f"Loaded {len(self.samples)} images from {data_dir}")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_transforms():
    """
    Define data augmentation and preprocessing transforms.
    
    Returns:
        Dictionary with 'train' and 'val' transforms
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    }
    return data_transforms


def main(args):
    """Main training function."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("WARNING: No GPU available. Training will be slow on CPU.")
        print("Consider training on a machine with GPU support.")
    
    # Create data transforms
    data_transforms = get_data_transforms()
    
    # Load datasets
    train_dataset = PostureDataset(
        data_dir=os.path.join(args.data_dir, 'train'),
        transform=data_transforms['train']
    )
    
    val_dataset = PostureDataset(
        data_dir=os.path.join(args.data_dir, 'val'),
        transform=data_transforms['val']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\nDataset Statistics:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Initialize model
    if args.model_type == 'deep':
        model = PostureCNNDeep(num_classes=2, dropout_rate=args.dropout)
        print("\nUsing Deep CNN architecture")
    else:
        model = PostureCNN(num_classes=2, dropout_rate=args.dropout)
        print("\nUsing Standard CNN architecture")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device
    )
    
    # Save final model
    output_path = args.output_path
    torch.save(trained_model.state_dict(), output_path)
    print(f"\nFinal model saved to: {output_path}")
    
    # Save model info
    model_info = {
        'model_type': args.model_type,
        'num_classes': 2,
        'classes': ['sitting_good', 'sitting_bad'],
        'input_size': (224, 224),
        'total_params': total_params,
        'epochs_trained': args.epochs,
    }
    
    info_path = output_path.replace('.pth', '_info.txt')
    with open(info_path, 'w') as f:
        for key, value in model_info.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Model info saved to: {info_path}")
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train CNN model for posture detection')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./data/training',
                       help='Root directory containing train/val subdirectories')
    parser.add_argument('--output_path', type=str, default='./data/inference_models/posture_cnn.pth',
                       help='Path to save trained model')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='standard',
                       choices=['standard', 'deep'],
                       help='Type of CNN architecture to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate for regularization')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate for optimizer')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    main(args)
