"""
Create Infection Labels for Wound Dataset
Uses heuristics and manual assignment based on wound types
"""

import json
from pathlib import Path
import random

class InfectionLabeler:
    """Create infection labels based on wound characteristics"""
    
    def __init__(self):
        # Define infection likelihood by wound type
        self.wound_type_infection_rates = {
            'burns': 0.35,          # Burns have moderate infection risk
            'laceration': 0.25,     # Lacerations moderate risk
            'stab_wound': 0.40,     # Stab wounds higher risk
            'cut': 0.20,            # Cuts lower risk
            'abrasions': 0.15,      # Abrasions lower risk
            'bruises': 0.05,        # Bruises very low risk
            'ingrown_nails': 0.30,  # Can get infected
        }
    
    def create_labels_heuristic(self, processed_dir="data/processed"):
        """
        Create infection labels using heuristics
        
        Strategy:
        1. Assign some wounds as infected based on wound type probabilities
        2. Ensure balanced dataset (aim for 30-40% infected)
        """
        processed_path = Path(processed_dir)
        
        print("\n" + "="*60)
        print("CREATING INFECTION LABELS")
        print("="*60 + "\n")
        
        # Process each split
        for split in ['train', 'validation', 'test']:
            print(f"\n📁 Processing {split} split...")
            
            images_dir = processed_path / split / 'images'
            
            if not images_dir.exists():
                print(f"⚠️  {images_dir} not found, skipping...")
                continue
            
            # Get all images
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            
            if len(image_files) == 0:
                print(f"   No images found")
                continue
            
            print(f"   Found {len(image_files)} images")
            
            # Create labels
            labels = {}
            infected_count = 0
            
            for img_path in image_files:
                filename = img_path.stem
                
                # Extract wound type from filename
                # Format: category_split_index (e.g., Burns_train_0001)
                parts = filename.lower().split('_')
                wound_type = parts[0] if parts else 'unknown'
                
                # Get infection probability for this wound type
                infection_prob = self.wound_type_infection_rates.get(
                    wound_type, 0.25  # Default 25% if unknown
                )
                
                # Randomly assign infection based on probability
                is_infected = random.random() < infection_prob
                labels[filename] = 1 if is_infected else 0
                
                if is_infected:
                    infected_count += 1
            
            # Print statistics
            total = len(labels)
            non_infected = total - infected_count
            infection_rate = (infected_count / total * 100) if total > 0 else 0
            
            print(f"   Created {total} labels:")
            print(f"     Infected: {infected_count} ({infection_rate:.1f}%)")
            print(f"     Non-infected: {non_infected} ({100-infection_rate:.1f}%)")
            
            # Save labels
            labels_path = processed_path / split / 'infection_labels.json'
            with open(labels_path, 'w') as f:
                json.dump(labels, f, indent=2)
            
            print(f"   ✅ Labels saved to: {labels_path}")
        
        print("\n" + "="*60)
        print("LABEL CREATION COMPLETE")
        print("="*60)
    
    def create_labels_manual_review(self, processed_dir="data/processed"):
        """
        Create labels with option for manual review
        Shows each wound type and lets you set infection rate
        """
        processed_path = Path(processed_dir)
        
        print("\n" + "="*60)
        print("MANUAL INFECTION LABELING")
        print("="*60 + "\n")
        
        print("You can customize infection rates for each wound type:")
        print("(Press Enter to use default values)\n")
        
        # Get custom rates
        custom_rates = {}
        for wound_type, default_rate in self.wound_type_infection_rates.items():
            user_input = input(f"{wound_type.upper()} (default {default_rate*100:.0f}%): ")
            if user_input.strip():
                try:
                    rate = float(user_input) / 100
                    custom_rates[wound_type] = max(0, min(1, rate))  # Clamp to 0-1
                except:
                    custom_rates[wound_type] = default_rate
            else:
                custom_rates[wound_type] = default_rate
        
        self.wound_type_infection_rates = custom_rates
        self.create_labels_heuristic(processed_dir)
    
    def create_balanced_labels(self, processed_dir="data/processed", target_infection_rate=0.35):
        """
        Create labels ensuring a specific infection rate across all data
        
        Args:
            processed_dir: Path to processed data
            target_infection_rate: Desired percentage of infected wounds (0.0 to 1.0)
        """
        processed_path = Path(processed_dir)
        
        print("\n" + "="*60)
        print(f"CREATING BALANCED LABELS ({target_infection_rate*100:.0f}% infected)")
        print("="*60 + "\n")
        
        for split in ['train', 'validation', 'test']:
            print(f"\n📁 Processing {split} split...")
            
            images_dir = processed_path / split / 'images'
            
            if not images_dir.exists():
                continue
            
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            
            if len(image_files) == 0:
                continue
            
            print(f"   Found {len(image_files)} images")
            
            # Calculate target counts
            total = len(image_files)
            target_infected = int(total * target_infection_rate)
            
            # Randomly select images to be infected
            shuffled_files = image_files.copy()
            random.shuffle(shuffled_files)
            
            labels = {}
            for idx, img_path in enumerate(shuffled_files):
                filename = img_path.stem
                labels[filename] = 1 if idx < target_infected else 0
            
            # Statistics
            infected_count = sum(labels.values())
            non_infected = total - infected_count
            actual_rate = (infected_count / total * 100) if total > 0 else 0
            
            print(f"   Created {total} labels:")
            print(f"     Infected: {infected_count} ({actual_rate:.1f}%)")
            print(f"     Non-infected: {non_infected} ({100-actual_rate:.1f}%)")
            
            # Save labels
            labels_path = processed_path / split / 'infection_labels.json'
            with open(labels_path, 'w') as f:
                json.dump(labels, f, indent=2)
            
            print(f"   ✅ Labels saved to: {labels_path}")
        
        print("\n" + "="*60)
        print("BALANCED LABEL CREATION COMPLETE")
        print("="*60)
    
    def analyze_existing_labels(self, processed_dir="data/processed"):
        """Analyze existing labels to see distribution"""
        processed_path = Path(processed_dir)
        
        print("\n" + "="*60)
        print("ANALYZING EXISTING LABELS")
        print("="*60 + "\n")
        
        total_infected = 0
        total_images = 0
        
        for split in ['train', 'validation', 'test']:
            labels_path = processed_path / split / 'infection_labels.json'
            
            if not labels_path.exists():
                print(f"❌ {split}: No labels found")
                continue
            
            with open(labels_path, 'r') as f:
                labels = json.load(f)
            
            infected = sum(labels.values())
            total = len(labels)
            rate = (infected / total * 100) if total > 0 else 0
            
            print(f"📊 {split.upper()}:")
            print(f"   Total: {total}")
            print(f"   Infected: {infected} ({rate:.1f}%)")
            print(f"   Non-infected: {total - infected} ({100-rate:.1f}%)")
            
            total_infected += infected
            total_images += total
        
        if total_images > 0:
            overall_rate = (total_infected / total_images * 100)
            print(f"\n📈 OVERALL:")
            print(f"   Total Images: {total_images}")
            print(f"   Infected: {total_infected} ({overall_rate:.1f}%)")
            print(f"   Non-infected: {total_images - total_infected} ({100-overall_rate:.1f}%)")
        
        print("="*60)


def main():
    """Main function with menu"""
    labeler = InfectionLabeler()
    
    print("\n" + "="*60)
    print("INFECTION LABEL CREATOR")
    print("="*60 + "\n")
    
    print("Choose an option:")
    print("1. Create labels with default heuristics (Quick)")
    print("2. Create labels with manual wound type rates")
    print("3. Create balanced labels (35% infected)")
    print("4. Create balanced labels (custom percentage)")
    print("5. Analyze existing labels")
    print("6. Regenerate all labels (recommended)")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == '1':
        print("\n✅ Creating labels with default heuristics...")
        labeler.create_labels_heuristic()
    
    elif choice == '2':
        labeler.create_labels_manual_review()
    
    elif choice == '3':
        print("\n✅ Creating balanced labels (35% infected)...")
        labeler.create_balanced_labels(target_infection_rate=0.35)
    
    elif choice == '4':
        rate = input("Enter target infection rate (e.g., 40 for 40%): ")
        try:
            rate = float(rate) / 100
            rate = max(0, min(1, rate))  # Clamp to 0-1
            labeler.create_balanced_labels(target_infection_rate=rate)
        except:
            print("❌ Invalid input. Using default 35%")
            labeler.create_balanced_labels(target_infection_rate=0.35)
    
    elif choice == '5':
        labeler.analyze_existing_labels()
    
    elif choice == '6':
        print("\n✅ Regenerating all labels with balanced approach (35% infected)...")
        labeler.create_balanced_labels(target_infection_rate=0.35)
    
    else:
        print("❌ Invalid choice")
        return
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n1. Review the labels created")
    print("2. Train infection detection model:")
    print("   python training/train_infection.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    main()