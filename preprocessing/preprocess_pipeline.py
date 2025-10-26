import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class WoundImagePreprocessor:
    def __init__(self, 
                 input_dir="data/raw",
                 output_dir="data/processed",
                 target_size=(512, 512),
                 quality_threshold=100):
        """
        Initialize preprocessor
        
        Args:
            input_dir: Directory containing raw images
            output_dir: Directory for processed images
            target_size: Target image dimensions (height, width)
            quality_threshold: Minimum Laplacian variance for blur detection
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.quality_threshold = quality_threshold
        
        # Create output directories
        self.create_output_structure()
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'processed': 0,
            'rejected_blur': 0,
            'rejected_format': 0,
            'rejected_size': 0
        }
    
    def create_output_structure(self):
        """Create organized folder structure"""
        splits = ['train', 'validation', 'test']
        subdirs = ['images', 'masks']
        
        for split in splits:
            for subdir in subdirs:
                path = self.output_dir / split / subdir
                path.mkdir(parents=True, exist_ok=True)
    
    def check_image_quality(self, image):
        """
        Check if image is blurry using Laplacian variance
        
        Args:
            image: numpy array (BGR)
        
        Returns:
            bool: True if image quality is acceptable
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var >= self.quality_threshold
    
    def check_brightness(self, image):
        """Check if image is too dark or too bright"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        return 20 < mean_brightness < 235  # Not too dark or bright
    
    def normalize_image(self, image):
        """
        Normalize image pixel values to [0, 1]
        
        Args:
            image: numpy array (BGR, 0-255)
        
        Returns:
            numpy array: normalized image (0-1)
        """
        return image.astype(np.float32) / 255.0
    
    def resize_image(self, image, mask=None):
        """
        Resize image and optional mask to target size
        
        Args:
            image: numpy array
            mask: numpy array (optional)
        
        Returns:
            resized image, resized mask (if provided)
        """
        resized_img = cv2.resize(image, self.target_size, 
                                 interpolation=cv2.INTER_AREA)
        
        if mask is not None:
            resized_mask = cv2.resize(mask, self.target_size, 
                                     interpolation=cv2.INTER_NEAREST)
            return resized_img, resized_mask
        
        return resized_img
    
    def load_image(self, image_path):
        """
        Load image from file, handle multiple formats
        
        Args:
            image_path: Path to image
        
        Returns:
            numpy array or None if failed
        """
        try:
            # Try OpenCV first
            image = cv2.imread(str(image_path))
            if image is not None:
                return image
            
            # Try PIL for other formats
            pil_image = Image.open(image_path)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return image
        
        except Exception as e:
            print(f"Error loading {image_path}: {str(e)}")
            return None
    
    def process_single_image(self, image_path, mask_path=None):
        """
        Process a single image through the pipeline
        
        Args:
            image_path: Path to image
            mask_path: Path to mask (optional)
        
        Returns:
            dict: processed data or None if rejected
        """
        self.stats['total_images'] += 1
        
        # Load image
        image = self.load_image(image_path)
        if image is None:
            self.stats['rejected_format'] += 1
            return None
        
        # Check minimum size
        h, w = image.shape[:2]
        if h < 128 or w < 128:
            self.stats['rejected_size'] += 1
            return None
        
        # Quality checks
        if not self.check_image_quality(image):
            self.stats['rejected_blur'] += 1
            return None
        
        if not self.check_brightness(image):
            self.stats['rejected_blur'] += 1
            return None
        
        # Load mask if provided
        mask = None
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Resize
        if mask is not None:
            image, mask = self.resize_image(image, mask)
        else:
            image = self.resize_image(image)
        
        # Normalize image
        image_normalized = self.normalize_image(image)
        
        self.stats['processed'] += 1
        
        return {
            'image': image,  # Keep original scale for saving
            'image_normalized': image_normalized,
            'mask': mask,
            'original_path': str(image_path)
        }
    
    def save_processed_data(self, data, filename, split='train'):
        """Save processed image and mask"""
        # Save image
        img_path = self.output_dir / split / 'images' / f"{filename}.jpg"
        cv2.imwrite(str(img_path), data['image'])
        
        # Save mask if available
        if data['mask'] is not None:
            mask_path = self.output_dir / split / 'masks' / f"{filename}.png"
            cv2.imwrite(str(mask_path), data['mask'])
    
    def balance_dataset(self, image_files, labels, target_ratio=1.0):
        """
        Balance dataset by oversampling minority class
        
        Args:
            image_files: list of image paths
            labels: list of labels (0 or 1 for binary)
            target_ratio: desired ratio of minority to majority class
        
        Returns:
            balanced image_files, balanced labels
        """
        from collections import Counter
        
        # Count classes
        class_counts = Counter(labels)
        majority_class = max(class_counts, key=class_counts.get)
        minority_class = min(class_counts, key=class_counts.get)
        
        majority_count = class_counts[majority_class]
        minority_count = class_counts[minority_class]
        
        print(f"Class distribution: {class_counts}")
        
        # Calculate target minority count
        target_minority = int(majority_count * target_ratio)
        oversample_count = target_minority - minority_count
        
        if oversample_count <= 0:
            return image_files, labels
        
        # Find minority samples
        minority_indices = [i for i, label in enumerate(labels) 
                           if label == minority_class]
        
        # Randomly oversample
        oversample_indices = np.random.choice(
            minority_indices, 
            size=oversample_count, 
            replace=True
        )
        
        # Combine
        balanced_files = image_files + [image_files[i] for i in oversample_indices]
        balanced_labels = labels + [labels[i] for i in oversample_indices]
        
        print(f"After balancing: {Counter(balanced_labels)}")
        
        return balanced_files, balanced_labels
    
    def split_dataset(self, image_files, labels=None, 
                     train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Split dataset into train/validation/test
        
        Args:
            image_files: list of image paths
            labels: list of labels (optional, for stratified split)
            train_ratio: proportion for training
            val_ratio: proportion for validation
            test_ratio: proportion for testing
        
        Returns:
            dict with train, validation, test splits
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        # First split: train vs (val + test)
        if labels is not None:
            train_files, temp_files, train_labels, temp_labels = train_test_split(
                image_files, labels, 
                test_size=(1 - train_ratio),
                stratify=labels,
                random_state=42
            )
        else:
            train_files, temp_files = train_test_split(
                image_files,
                test_size=(1 - train_ratio),
                random_state=42
            )
            train_labels, temp_labels = None, None
        
        # Second split: val vs test
        val_test_ratio = test_ratio / (val_ratio + test_ratio)
        
        if temp_labels is not None:
            val_files, test_files, val_labels, test_labels = train_test_split(
                temp_files, temp_labels,
                test_size=val_test_ratio,
                stratify=temp_labels,
                random_state=42
            )
        else:
            val_files, test_files = train_test_split(
                temp_files,
                test_size=val_test_ratio,
                random_state=42
            )
            val_labels, test_labels = None, None
        
        return {
            'train': {'files': train_files, 'labels': train_labels},
            'validation': {'files': val_files, 'labels': val_labels},
            'test': {'files': test_files, 'labels': test_labels}
        }
    
    def process_directory(self, dataset_name, has_masks=False, has_labels=False):
        """
        Process entire directory of images
        
        Args:
            dataset_name: name of dataset folder in input_dir
            has_masks: whether dataset includes segmentation masks
            has_labels: whether dataset includes classification labels
        """
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}\n")
        
        dataset_path = self.input_dir / dataset_name
        
        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(dataset_path.glob(f'**/*{ext}')))
            image_files.extend(list(dataset_path.glob(f'**/*{ext.upper()}')))
        
        print(f"Found {len(image_files)} images")
        
        if len(image_files) == 0:
            print("⚠ No images found!")
            return
        
        # Load labels if available
        labels = None
        if has_labels:
            labels_file = dataset_path / 'labels.json'
            if labels_file.exists():
                with open(labels_file, 'r') as f:
                    labels_dict = json.load(f)
                labels = [labels_dict.get(Path(f).stem, 0) for f in image_files]
        
        # Split dataset
        splits = self.split_dataset(image_files, labels)
        
        # Process each split
        for split_name, split_data in splits.items():
            print(f"\nProcessing {split_name} split...")
            files = split_data['files']
            
            for idx, img_path in enumerate(tqdm(files)):
                # Find corresponding mask
                mask_path = None
                if has_masks:
                    mask_name = img_path.stem + '_mask.png'
                    mask_path = img_path.parent / mask_name
                    if not mask_path.exists():
                        mask_path = None
                
                # Process image
                result = self.process_single_image(img_path, mask_path)
                
                if result is not None:
                    # Generate unique filename
                    filename = f"{dataset_name}_{split_name}_{idx:04d}"
                    self.save_processed_data(result, filename, split_name)
        
        # Print statistics
        self.print_statistics()
    
    def print_statistics(self):
        """Print processing statistics"""
        print(f"\n{'='*60}")
        print("PREPROCESSING STATISTICS")
        print(f"{'='*60}")
        print(f"Total images found: {self.stats['total_images']}")
        print(f"Successfully processed: {self.stats['processed']}")
        print(f"Rejected (blur/brightness): {self.stats['rejected_blur']}")
        print(f"Rejected (format error): {self.stats['rejected_format']}")
        print(f"Rejected (size too small): {self.stats['rejected_size']}")
        
        success_rate = (self.stats['processed'] / self.stats['total_images'] * 100 
                       if self.stats['total_images'] > 0 else 0)
        print(f"\nSuccess rate: {success_rate:.1f}%")
    
    def create_dataset_manifest(self):
        """Create JSON manifest of processed dataset"""
        manifest = {
            'target_size': self.target_size,
            'splits': {}
        }
        
        for split in ['train', 'validation', 'test']:
            img_dir = self.output_dir / split / 'images'
            mask_dir = self.output_dir / split / 'masks'
            
            images = list(img_dir.glob('*.jpg'))
            masks = list(mask_dir.glob('*.png'))
            
            manifest['splits'][split] = {
                'num_images': len(images),
                'num_masks': len(masks),
                'image_dir': str(img_dir),
                'mask_dir': str(mask_dir)
            }
        
        manifest_path = self.output_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n✓ Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = WoundImagePreprocessor(
        input_dir="data/raw",
        output_dir="data/processed",
        target_size=(512, 512),
        quality_threshold=100
    )
    
    # Process sample dataset
    print("Starting preprocessing pipeline...")
    
    # Example: Process your dataset
    # preprocessor.process_directory('sample', has_masks=True, has_labels=False)
    
    # Create manifest
    preprocessor.create_dataset_manifest()
    
    print("\n✓ Preprocessing complete!")