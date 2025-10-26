import sys

def verify_installation():
    """Verify all dependencies are installed correctly"""
    
    print("Checking Python dependencies...\n")
    
    dependencies = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'tensorflow': 'TensorFlow',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'PIL': 'Pillow',
        'albumentations': 'Albumentations',
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn',
        'matplotlib': 'Matplotlib',
        'reportlab': 'ReportLab',
    }
    
    missing = []
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - MISSING")
            missing.append(name)
    
    print("\n" + "="*50)
    
    if missing:
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All Python dependencies installed successfully!")
        
        # Check CUDA availability
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ CUDA not available (CPU mode)")
        
        return True

if __name__ == "__main__":
    success = verify_installation()
    sys.exit(0 if success else 1)