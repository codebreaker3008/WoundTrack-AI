"""
Training Script for Wound Segmentation Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.segmentation_model import UNet, CombinedLoss, calculate_metrics


class WoundSegmentationDataset(Dataset):
    """Dataset for wound segmentation"""
    
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        
        # Get all image files
        self.image_files = sorted(list(self.image_dir.glob('*.jpg')) + 
                                 list(self.image_dir.glob('*.png')))
        
        print(f"Found {len(self.image_files)} images in {image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_path = self.mask_dir / (img_path.stem + '.png')
        if not mask_path.exists():
            # Try alternative naming
            mask_path = self.mask_dir / (img_path.stem + '_mask.png')
        
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # Create empty mask if not found
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Normalize mask to 0-1
        mask = (mask > 0).astype(np.float32)
        
        # Apply transforms if any
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        
        return image, mask


class SegmentationTrainer:
    """Trainer for segmentation model"""
    
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader,
                 criterion,
                 optimizer,
                 scheduler=None,
                 device='cuda',
                 save_dir='models/saved_models/segmentation'):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_iou': [],
            'val_dice': [],
            'val_accuracy': []
        }
        
        self.best_val_loss = float('inf')
        self.best_val_dice = 0.0
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        all_metrics = {
            'iou': [],
            'dice': [],
            'accuracy': [],
            'precision': [],
            'recall': []
        }
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                running_loss += loss.item()
                
                # Calculate metrics
                metrics = calculate_metrics(outputs, masks)
                for key in all_metrics:
                    all_metrics[key].append(metrics[key])
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Average metrics
        epoch_loss = running_loss / len(self.val_loader)
        avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        
        return epoch_loss, avg_metrics
    
    def train(self, num_epochs, early_stopping_patience=10):
        """Train the model"""
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate(epoch)
            self.history['val_loss'].append(val_loss)
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val IoU: {val_metrics['iou']:.4f}")
            print(f"Val Dice: {val_metrics['dice']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Save best model (based on Dice score)
            if val_metrics['dice'] > self.best_val_dice:
                self.best_val_dice = val_metrics['dice']
                self.save_checkpoint(epoch, 'best_dice_model.pth')
                print(f"✓ New best Dice score: {self.best_val_dice:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save best model (based on loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, 'best_loss_model.pth')
                print(f"✓ New best validation loss: {self.best_val_loss:.4f}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pth')
            
            print()
        
        # Save final model
        self.save_checkpoint(num_epochs, 'final_model.pth')
        
        # Save training history
        self.save_history()
        
        # Plot training curves
        self.plot_training_curves()
        
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Best Dice Score: {self.best_val_dice:.4f}")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
    
    def save_checkpoint(self, epoch, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_dice': self.best_val_dice,
            'history': self.history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
        # print(f"Checkpoint saved: {save_path}")
    
    def save_history(self):
        """Save training history to JSON"""
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved: {history_path}")
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Dice Score
        axes[0, 1].plot(self.history['val_dice'], label='Val Dice', color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].set_title('Validation Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # IoU
        axes[1, 0].plot(self.history['val_iou'], label='Val IoU', color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU')
        axes[1, 0].set_title('Validation IoU')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Accuracy
        axes[1, 1].plot(self.history['val_accuracy'], label='Val Accuracy', color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Validation Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.save_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved: {plot_path}")
        plt.close()


def main():
    """Main training function"""
    
    # Configuration
    config = {
        'data_dir': 'data/processed',
        'batch_size': 8,
        'num_epochs': 100,
        'learning_rate': 1e-4,
        'image_size': 512,
        'early_stopping_patience': 15,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Configuration: {json.dumps(config, indent=2)}\n")
    
    # Create datasets
    train_dataset = WoundSegmentationDataset(
        image_dir=f"{config['data_dir']}/train/images",
        mask_dir=f"{config['data_dir']}/train/masks"
    )
    
    val_dataset = WoundSegmentationDataset(
        image_dir=f"{config['data_dir']}/validation/images",
        mask_dir=f"{config['data_dir']}/validation/masks"
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True if config['device'] == 'cuda' else False
    )
    
    # Create model
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    
    # Loss function
    criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Create trainer
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config['device']
    )
    
    # Train
    trainer.train(
        num_epochs=config['num_epochs'],
        early_stopping_patience=config['early_stopping_patience']
    )


if __name__ == "__main__":
    main()