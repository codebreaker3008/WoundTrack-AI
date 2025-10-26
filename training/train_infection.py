"""
Training Script for Wound Infection Detection Model
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
from sklearn.metrics import roc_auc_score, confusion_matrix
import seaborn as sns
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.infection_model import (
    InfectionDetectionModel, 
    WeightedBCELoss, 
    FocalLoss,
    calculate_classification_metrics,
    get_data_transforms
)


class WoundInfectionDataset(Dataset):
    """Dataset for wound infection classification"""
    
    def __init__(self, image_dir, labels_file, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Load labels
        with open(labels_file, 'r') as f:
            self.labels_dict = json.load(f)
        
        # Get image files that have labels
        self.samples = []
        for img_path in self.image_dir.glob('*.jpg'):
            if img_path.stem in self.labels_dict:
                label = self.labels_dict[img_path.stem]
                self.samples.append((img_path, label))
        
        print(f"Found {len(self.samples)} labeled images")
        
        # Count class distribution
        labels = [label for _, label in self.samples]
        self.class_counts = {
            'infected': sum(labels),
            'non_infected': len(labels) - sum(labels)
        }
        print(f"Class distribution: {self.class_counts}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            from PIL import Image
            image = Image.fromarray(image)
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        label = torch.tensor([label], dtype=torch.float32)
        
        return image, label


class InfectionTrainer:
    """Trainer for infection detection model"""
    
    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 criterion,
                 optimizer,
                 scheduler=None,
                 device='cuda',
                 save_dir='models/saved_models/infection_detection'):
        
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
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_auc': []
        }
        
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
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
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                
                # Collect predictions
                probs = torch.sigmoid(outputs)
                all_probs.extend(probs.cpu().numpy().flatten())
                all_predictions.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        epoch_loss = running_loss / len(self.val_loader)
        
        # Convert to tensors for metrics calculation
        pred_tensor = torch.tensor(all_predictions).unsqueeze(1)
        label_tensor = torch.tensor(all_labels).unsqueeze(1)
        metrics = calculate_classification_metrics(pred_tensor, label_tensor)
        
        # Calculate AUC
        try:
            auc = roc_auc_score(all_labels, all_probs)
            metrics['auc'] = auc
        except:
            metrics['auc'] = 0.0
        
        return epoch_loss, metrics
    
    def train(self, num_epochs, early_stopping_patience=15):
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
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_auc'].append(val_metrics.get('auc', 0.0))
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}")
            print(f"Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            print(f"Val AUC: {val_metrics.get('auc', 0.0):.4f}")
            
            # Save best model based on F1 score
            if val_metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1']
                self.save_checkpoint(epoch, 'best_f1_model.pth')
                print(f"✓ New best F1 score: {self.best_val_f1:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save best model based on loss
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
        
        # Plot confusion matrix
        self.plot_confusion_matrix()
        
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Best F1 Score: {self.best_val_f1:.4f}")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
    
    def save_checkpoint(self, epoch, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1,
            'history': self.history
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
    
    def save_history(self):
        """Save training history to JSON"""
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved: {history_path}")
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.history['val_accuracy'], color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].grid(True)
        
        # F1 Score
        axes[0, 2].plot(self.history['val_f1'], color='purple')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('F1 Score')
        axes[0, 2].set_title('Validation F1 Score')
        axes[0, 2].grid(True)
        
        # Precision
        axes[1, 0].plot(self.history['val_precision'], color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Validation Precision')
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.history['val_recall'], color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_title('Validation Recall')
        axes[1, 1].grid(True)
        
        # AUC
        axes[1, 2].plot(self.history['val_auc'], color='blue')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('AUC')
        axes[1, 2].set_title('Validation AUC-ROC')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.save_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved: {plot_path}")
        plt.close()
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix on validation set"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                all_predictions.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Infected', 'Infected'],
                   yticklabels=['Non-Infected', 'Infected'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix - Validation Set')
        
        # Save
        cm_path = self.save_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved: {cm_path}")
        plt.close()


def calculate_class_weights(dataset):
    """Calculate class weights for imbalanced dataset"""
    total = len(dataset)
    class_counts = dataset.class_counts
    
    # Calculate weights (inverse of frequency)
    weight_infected = total / (2 * class_counts['infected'])
    weight_non_infected = total / (2 * class_counts['non_infected'])
    
    print(f"\nClass weights calculated:")
    print(f"Non-Infected: {weight_non_infected:.4f}")
    print(f"Infected: {weight_infected:.4f}")
    
    return weight_infected


def main():
    """Main training function"""
    
    # Configuration
    config = {
        'data_dir': 'data/processed',
        'labels_file': 'data/processed/infection_labels.json',
        'batch_size': 16,
        'num_epochs': 50,
        'learning_rate': 1e-4,
        'image_size': 224,
        'model_type': 'resnet50',  # or 'efficientnet_b0'
        'use_focal_loss': False,  # Set True for heavily imbalanced data
        'early_stopping_patience': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Configuration: {json.dumps(config, indent=2)}\n")
    
    # Create transforms
    train_transform = get_data_transforms(config['image_size'], mode='train')
    val_transform = get_data_transforms(config['image_size'], mode='val')
    
    # Create datasets
    train_dataset = WoundInfectionDataset(
        image_dir=f"{config['data_dir']}/train/images",
        labels_file=f"{config['data_dir']}/train/infection_labels.json",
        transform=train_transform
    )
    
    val_dataset = WoundInfectionDataset(
        image_dir=f"{config['data_dir']}/validation/images",
        labels_file=f"{config['data_dir']}/validation/infection_labels.json",
        transform=val_transform
    )
    
    # Calculate class weights
    pos_weight = calculate_class_weights(train_dataset)
    
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
    model = InfectionDetectionModel(
        num_classes=1,
        pretrained=True,
        model_type=config['model_type']
    )
    
    # Freeze backbone initially for faster training
    print("\nFreezing backbone for initial training...")
    model.freeze_backbone()
    
    # Loss function
    if config['use_focal_loss']:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        criterion = WeightedBCELoss(
            pos_weight=torch.tensor([pos_weight])
        )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
    )
    
    # Create trainer
    trainer = InfectionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config['device']
    )
    
    # Train with frozen backbone
    print("\n" + "="*60)
    print("PHASE 1: Training with frozen backbone")
    print("="*60)
    trainer.train(
        num_epochs=config['num_epochs'] // 2,
        early_stopping_patience=config['early_stopping_patience']
    )
    
    # Unfreeze and fine-tune
    print("\n" + "="*60)
    print("PHASE 2: Fine-tuning entire model")
    print("="*60)
    model.unfreeze_backbone()
    
    # Lower learning rate for fine-tuning
    for param_group in optimizer.param_groups:
        param_group['lr'] = config['learning_rate'] / 10
    
    trainer.train(
        num_epochs=config['num_epochs'] // 2,
        early_stopping_patience=config['early_stopping_patience']
    )


if __name__ == "__main__":
    main()