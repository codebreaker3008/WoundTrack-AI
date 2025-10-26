import os
import torch
import torchvision.models as models

def download_pretrained_models():
    """Download pre-trained models for transfer learning"""
    
    models_dir = "models/saved_models"
    os.makedirs(models_dir, exist_ok=True)
    
    print("Downloading pre-trained models...")
    
    # ResNet50 for infection detection
    print("1. Downloading ResNet50...")
    resnet50 = models.resnet50(pretrained=True)
    torch.save(resnet50.state_dict(), 
               f"{models_dir}/resnet50_pretrained.pth")
    
    # EfficientNet-B0 for infection detection
    print("2. Downloading EfficientNet-B0...")
    efficientnet = models.efficientnet_b0(pretrained=True)
    torch.save(efficientnet.state_dict(), 
               f"{models_dir}/efficientnet_b0_pretrained.pth")
    
    # EfficientNet-B3 for healing stage classification
    print("3. Downloading EfficientNet-B3...")
    efficientnet_b3 = models.efficientnet_b3(pretrained=True)
    torch.save(efficientnet_b3.state_dict(), 
               f"{models_dir}/efficientnet_b3_pretrained.pth")
    
    print("✓ All pre-trained models downloaded successfully!")

if __name__ == "__main__":
    download_pretrained_models()