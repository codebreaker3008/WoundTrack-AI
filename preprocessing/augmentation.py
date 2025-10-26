"""
Advanced Data Augmentation for Wound Images
Generates realistic variations while maintaining medical validity
"""

import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

class WoundAugmentation:
    def __init__(self, target_size=(512, 512)):
        """Initialize augmentation pipeline"""
        self.target_size = target_size
        
        # Training augmentation pipeline
        self.train_transform = A.Compose([
            # Geometric transformations
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=45,
                p=0.5
            ),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=0.3
            ),
            
            # Color/brightness augmentations
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0
                ),
                A.RGBShift(
                    r_shift_limit=15,
                    g_shift_limit=15,
                    b_shift_limit=15,
                    p=1.0
                ),
            ], p=0.8),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            
            # Simulate different lighting conditions
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_lower=1,
                num_shadows_upper=2,
                shadow_dimension=5,
                p=0.2
            ),
            
            # Simulate different camera quality
            A.OneOf([
                A.ImageCompression(quality_lower=75, quality_upper=100, p=1.0),
                A.Downscale(scale_min=0.75, scale_max=0.95, p=1.0),
            ], p=0.2),
            
            # Normalize
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
        ])
        
        # Validation/test transform (no augmentation)
        self.val_transform = A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),
        ])
        
        # Heavy augmentation for generating synthetic samples
        self.heavy_augment = A.Compose([
            A.RandomRotate90(p=1.0),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=180,
                p=0.8
            ),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=0.5
            ),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=0.3),
            
            # Aggressive color changes
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.8
            ),
            A.HueSaturationValue(
                hue_shift_limit=30,
                sat_shift_limit=40,
                val_shift_limit=30,
                p=0.8
            ),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
            A.ChannelShuffle(p=0.2),
            
            # Noise
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 100.0), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            ], p=0.5),
            
            # Blur
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ], p=0.4),
            
            A.RandomShadow(p=0.3),
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
        ])
    
    def simulate_skin_tone_variation(self, image):
        """
        Simulate different skin tones
        """
        # Define skin tone adjustments (RGB shifts)
        skin_tones = [
            {'r': 0, 'g': 0, 'b': 0},      # Original
            {'r': 20, 'g': 10, 'b': -10},   # Lighter
            {'r': -15, 'g': -20, 'b': -5},  # Darker
            {'r': 10, 'g': 5, 'b': -5},     # Warmer
            {'r': -5, 'g': 0, 'b': 5},      # Cooler
        ]
        
        tone = np.random.choice(len(skin_tones))
        shifts = skin_tones[tone]
        
        adjusted = image.copy().astype(np.float32)
        adjusted[:, :, 2] += shifts['r']  # R
        adjusted[:, :, 1] += shifts['g']  # G
        adjusted[:, :, 0] += shifts['b']  # B
        
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        return adjusted
    
    def augment_with_mask(self, image, mask):
        """
        Apply augmentation to both image and mask
        Ensures geometric transforms are applied consistently
        """
        transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=45,
                p=0.5
            ),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=0.3
            ),
        ])
        
        augmented = transform(image=image, mask=mask)
        return augmented['image'], augmented['mask']
    
    def generate_augmented_dataset(self, 
                                   input_dir, 
                                   output_dir, 
                                   num_augmentations=10,
                                   has_masks=True):
        """
        Generate augmented versions of dataset
        
        Args:
            input_dir: Directory with original images
            output_dir: Directory to save augmented images
            num_augmentations: Number of augmented versions per image
            has_masks: Whether masks are available
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if has_masks:
            (output_path / 'images').mkdir(exist_ok=True)
            (output_path / 'masks').mkdir(exist_ok=True)
        
        # Find all images
        image_files = list(input_path.glob('images/*.jpg')) + \
                     list(input_path.glob('images/*.png'))
        
        print(f"Generating {num_augmentations} augmentations for {len(image_files)} images...")
        
        for img_path in tqdm(image_files):
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Load mask if available
            mask = None
            if has_masks:
                mask_path = input_path / 'masks' / (img_path.stem + '.png')
                if mask_path.exists():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
            # Generate augmentations
            for aug_idx in range(num_augmentations):
                # Apply heavy augmentation
                if mask is not None:
                    # Apply geometric transforms to both
                    aug_img, aug_mask = self.augment_with_mask(image, mask)
                    
                    # Apply color transforms only to image
                    color_transform = A.Compose([
                        A.RandomBrightnessContrast(p=0.8),
                        A.HueSaturationValue(p=0.8),
                        A.GaussNoise(p=0.3),
                    ])
                    aug_img = color_transform(image=aug_img)['image']
                    
                    # Simulate skin tone variation
                    if np.random.random() > 0.5:
                        aug_img = self.simulate_skin_tone_variation(aug_img)
                else:
                    aug_img = self.heavy_augment(image=image)['image']
                
                # Save augmented image
                aug_filename = f"{img_path.stem}_aug_{aug_idx:03d}.jpg"
                if has_masks:
                    save_path = output_path / 'images' / aug_filename
                else:
                    save_path = output_path / aug_filename
                cv2.imwrite(str(save_path), aug_img)
                
                # Save augmented mask
                if mask is not None and aug_mask is not None:
                    mask_filename = f"{img_path.stem}_aug_{aug_idx:03d}.png"
                    mask_save_path = output_path / 'masks' / mask_filename
                    cv2.imwrite(str(mask_save_path), aug_mask)
        
        print(f"\n✓ Augmentation complete! Saved to: {output_path}")
    
    def preview_augmentations(self, image_path, mask_path=None, num_samples=6):
        """
        Preview augmentation results
        
        Args:
            image_path: Path to sample image
            mask_path: Path to sample mask (optional)
            num_samples: Number of augmentation samples to show
        """
        # Load image
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask if provided
        mask = None
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Create figure
        rows = (num_samples + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
        axes = axes.flatten()
        
        # Show original
        axes[0].imshow(image_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Generate and show augmentations
        for idx in range(1, num_samples):
            if mask is not None:
                aug_img, aug_mask = self.augment_with_mask(image, mask)
                
                # Apply additional color transforms
                color_transform = A.Compose([
                    A.RandomBrightnessContrast(p=1.0),
                    A.HueSaturationValue(p=1.0),
                ])
                aug_img = color_transform(image=aug_img)['image']
                
                # Simulate skin tone
                if idx % 2 == 0:
                    aug_img = self.simulate_skin_tone_variation(aug_img)
                
                # Overlay mask on image for visualization
                aug_img_rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
                mask_colored = np.zeros_like(aug_img_rgb)
                mask_colored[:, :, 0] = aug_mask  # Red channel
                overlay = cv2.addWeighted(aug_img_rgb, 0.7, mask_colored, 0.3, 0)
                
                axes[idx].imshow(overlay)
            else:
                aug_img = self.heavy_augment(image=image)['image']
                aug_img = self.simulate_skin_tone_variation(aug_img)
                aug_img_rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
                axes[idx].imshow(aug_img_rgb)
            
            axes[idx].set_title(f'Augmentation {idx}')
            axes[idx].axis('off')
        
        # Hide unused subplots
        for idx in range(num_samples, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig('augmentation_preview.png', dpi=150, bbox_inches='tight')
        print("\n✓ Preview saved to: augmentation_preview.png")
        plt.show()
    
    def get_transform(self, mode='train'):
        """
        Get appropriate transform for given mode
        
        Args:
            mode: 'train', 'val', or 'test'
        
        Returns:
            Albumentations transform
        """
        if mode == 'train':
            return self.train_transform
        else:
            return self.val_transform


# PyTorch Dataset with Augmentation
class WoundDataset:
    """Custom dataset for wound images with augmentation"""
    
    def __init__(self, image_dir, mask_dir=None, transform=None):
        """
        Initialize dataset
        
        Args:
            image_dir: Directory containing images
            mask_dir: Directory containing masks (optional)
            transform: Albumentations transform
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.transform = transform
        
        # Get image files
        self.image_files = sorted(list(self.image_dir.glob('*.jpg')) + 
                                 list(self.image_dir.glob('*.png')))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask if available
        mask = None
        if self.mask_dir:
            mask_path = self.mask_dir / (img_path.stem + '.png')
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            else:
                augmented = self.transform(image=image)
                image = augmented['image']
        
        return {
            'image': image,
            'mask': mask,
            'filename': img_path.name
        }


if __name__ == "__main__":
    # Initialize augmentation
    augmenter = WoundAugmentation(target_size=(512, 512))
    
    # Example 1: Preview augmentations
    print("Example 1: Preview augmentations")
    print("Replace 'path/to/image.jpg' with actual image path")
    # augmenter.preview_augmentations(
    #     'path/to/image.jpg',
    #     'path/to/mask.png',
    #     num_samples=9
    # )
    
    # Example 2: Generate augmented dataset
    print("\nExample 2: Generate augmented dataset")
    # augmenter.generate_augmented_dataset(
    #     input_dir='data/processed/train',
    #     output_dir='data/augmented/train',
    #     num_augmentations=10,
    #     has_masks=True
    # )
    
    print("\n✓ Augmentation module ready!")
    print("Uncomment examples above and provide actual image paths to test.")