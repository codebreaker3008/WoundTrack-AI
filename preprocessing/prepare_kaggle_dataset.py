"""
Prepare Kaggle Wound Dataset for Training
This script processes the downloaded Kaggle dataset and prepares it for model training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.preprocess_pipeline import WoundImagePreprocessor
from pathlib import Path
import json
import shutil

def prepare_kaggle_dataset():
    """Prepare Kaggle wound dataset for training"""
    
    print("\n" + "="*60)
    print("KAGGLE WOUND DATASET PREPARATION")
    print("="*60 + "\n")
    
    # Paths
    kaggle_data_path = Path("data/raw/kaggle_wounds/organized")
    
    if not kaggle_data_path.exists():
        print("❌ Organized Kaggle dataset not found!")
        print(f"   Expected path: {kaggle_data_path}")
        print("\n💡 Run this first:")
        print("   python preprocessing/download_kaggle_dataset.py")
        return False
    
    print(f"✅ Found Kaggle dataset at: {kaggle_data_path}")
    
    # Initialize preprocessor
    preprocessor = WoundImagePreprocessor(
        input_dir=str(kaggle_data_path),
        output_dir="data/processed",
        target_size=(512, 512),
        quality_threshold=100
    )
    
    # Find all subdirectories with images
    categories = [d for d in kaggle_data_path.iterdir() if d.is_dir()]
    
    print(f"\n📊 Found {len(categories)} categories:")
    for cat in categories:
        img_count = len(list(cat.glob("*.jpg"))) + len(list(cat.glob("*.png")))
        print(f"   {cat.name}: {img_count} images")
    
    # Process all images
    print("\n" + "="*60)
    print("PREPROCESSING IMAGES")
    print("="*60 + "\n")
    
    all_processed = []
    
    for category_dir in categories:
        category_name = category_dir.name
        print(f"\n📁 Processing category: {category_name}")
        
        # Get all images in category
        image_files = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
        
        if len(image_files) == 0:
            print(f"   ⚠️  No images found, skipping...")
            continue
        
        # Load infection labels if available
        labels_file = kaggle_data_path / 'infection_labels.json'
        labels = {}
        if labels_file.exists():
            with open(labels_file, 'r') as f:
                labels = json.load(f)
        
        # Split into train/val/test
        from sklearn.model_selection import train_test_split
        
        # Get labels for this category
        category_labels = [labels.get(f.stem, 0) for f in image_files]
        
        # Split: 70% train, 15% val, 15% test
        train_files, temp_files, train_labels, temp_labels = train_test_split(
            image_files, category_labels, test_size=0.3, random_state=42,
            stratify=category_labels if len(set(category_labels)) > 1 else None
        )
        
        val_files, test_files, val_labels, test_labels = train_test_split(
            temp_files, temp_labels, test_size=0.5, random_state=42,
            stratify=temp_labels if len(set(temp_labels)) > 1 else None
        )
        
        print(f"   Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        
        # Process each split
        splits = {
            'train': (train_files, train_labels),
            'validation': (val_files, val_labels),
            'test': (test_files, test_labels)
        }
        
        for split_name, (files, split_labels) in splits.items():
            print(f"\n   Processing {split_name} split...")
            
            split_processed = []
            
            for idx, (img_path, label) in enumerate(zip(files, split_labels)):
                # Process image
                result = preprocessor.process_single_image(img_path)
                
                if result is not None:
                    # Generate filename
                    filename = f"{category_name}_{split_name}_{idx:04d}"
                    
                    # Save processed image
                    preprocessor.save_processed_data(result, filename, split_name)
                    
                    split_processed.append({
                        'filename': filename,
                        'category': category_name,
                        'infection_label': label,
                        'original_path': str(img_path)
                    })
            
            all_processed.extend(split_processed)
            
            # Save labels for this split
            split_labels_dict = {
                item['filename']: item['infection_label'] 
                for item in split_processed
            }
            
            labels_path = Path(f"data/processed/{split_name}/infection_labels.json")
            
            # Merge with existing labels if any
            if labels_path.exists():
                with open(labels_path, 'r') as f:
                    existing_labels = json.load(f)
                existing_labels.update(split_labels_dict)
                split_labels_dict = existing_labels
            
            with open(labels_path, 'w') as f:
                json.dump(split_labels_dict, f, indent=2)
    
    # Print statistics
    preprocessor.print_statistics()
    
    # Create manifest
    preprocessor.create_dataset_manifest()
    
    # Save processing summary
    summary = {
        'source': 'Kaggle Wound Dataset',
        'processed_images': len(all_processed),
        'categories': list(set([item['category'] for item in all_processed])),
        'infection_distribution': {
            'infected': sum([item['infection_label'] for item in all_processed]),
            'non_infected': len(all_processed) - sum([item['infection_label'] for item in all_processed])
        }
    }
    
    summary_path = Path("data/processed/processing_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"   Total processed: {summary['processed_images']}")
    print(f"   Categories: {', '.join(summary['categories'])}")
    print(f"   Infected: {summary['infection_distribution']['infected']}")
    print(f"   Non-infected: {summary['infection_distribution']['non_infected']}")
    print(f"\n📄 Summary saved: {summary_path}")
    
    print("\n💡 Next steps:")
    print("   1. Review processed data in data/processed/")
    print("   2. Run data augmentation (optional):")
    print("      python preprocessing/augmentation.py")
    print("   3. Start training:")
    print("      python training/train_infection.py")
    print("="*60 + "\n")
    
    return True


if __name__ == "__main__":
    prepare_kaggle_dataset()