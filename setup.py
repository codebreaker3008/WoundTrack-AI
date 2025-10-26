"""
Complete Setup and Verification Script for Wound Healing Tracker
Run this script to set up the entire project
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import json

class Colors:
    """Terminal colors"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print colored header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")

def run_command(command, cwd=None, check=True):
    """Run shell command"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            check=check,
            capture_output=True,
            text=True
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr

def check_python_version():
    """Check Python version"""
    print_header("CHECKING PYTHON VERSION")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and 8 <= version.minor <= 11:
        print_success(f"Python version: {version_str}")
        return True
    else:
        print_error(f"Python version {version_str} not supported")
        print_info("Required: Python 3.8-3.11")
        return False

def check_system_dependencies():
    """Check system dependencies"""
    print_header("CHECKING SYSTEM DEPENDENCIES")
    
    dependencies = {
        'git': 'git --version',
        'node': 'node --version',
        'npm': 'npm --version'
    }
    
    all_present = True
    for name, command in dependencies.items():
        success, stdout, _ = run_command(command, check=False)
        if success:
            version = stdout.strip()
            print_success(f"{name}: {version}")
        else:
            print_error(f"{name}: Not found")
            all_present = False
    
    return all_present

def create_directory_structure():
    """Create project directory structure"""
    print_header("CREATING DIRECTORY STRUCTURE")
    
    directories = [
        'data/raw',
        'data/processed/train/images',
        'data/processed/train/masks',
        'data/processed/validation/images',
        'data/processed/validation/masks',
        'data/processed/test/images',
        'data/processed/test/masks',
        'data/augmented',
        'models/saved_models/segmentation',
        'models/saved_models/infection_detection',
        'models/saved_models/healing_stage',
        'uploads',
        'results',
        'logs',
        'notebooks',
        'tests',
        'docs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print_success(f"Created {len(directories)} directories")
    return True

def setup_python_environment():
    """Setup Python virtual environment"""
    print_header("SETTING UP PYTHON ENVIRONMENT")
    
    # Check if venv exists
    if Path('venv').exists():
        print_warning("Virtual environment already exists")
        response = input("Recreate it? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree('venv')
        else:
            print_info("Skipping virtual environment creation")
            return True
    
    # Create venv
    print_info("Creating virtual environment...")
    success, _, stderr = run_command(f"{sys.executable} -m venv venv")
    
    if success:
        print_success("Virtual environment created")
        return True
    else:
        print_error(f"Failed to create virtual environment: {stderr}")
        return False

def install_python_dependencies():
    """Install Python dependencies"""
    print_header("INSTALLING PYTHON DEPENDENCIES")
    
    # Determine pip path
    if sys.platform == 'win32':
        pip_path = 'venv\\Scripts\\pip'
    else:
        pip_path = 'venv/bin/pip'
    
    if not Path(pip_path).exists():
        print_error("Virtual environment not found. Run setup first.")
        return False
    
    # Upgrade pip
    print_info("Upgrading pip...")
    run_command(f"{pip_path} install --upgrade pip")
    
    # Install from requirements.txt
    if Path('requirements.txt').exists():
        print_info("Installing dependencies from requirements.txt...")
        success, stdout, stderr = run_command(
            f"{pip_path} install -r requirements.txt"
        )
        
        if success:
            print_success("Python dependencies installed")
            return True
        else:
            print_error(f"Failed to install dependencies: {stderr}")
            return False
    else:
        print_error("requirements.txt not found")
        return False

def setup_frontend():
    """Setup frontend"""
    print_header("SETTING UP FRONTEND")
    
    if not Path('frontend').exists():
        print_error("Frontend directory not found")
        return False
    
    # Check if node_modules exists
    if Path('frontend/node_modules').exists():
        print_warning("node_modules already exists")
        response = input("Reinstall dependencies? (y/n): ")
        if response.lower() != 'y':
            print_info("Skipping frontend setup")
            return True
    
    # Install dependencies
    print_info("Installing frontend dependencies...")
    success, stdout, stderr = run_command('npm install', cwd='frontend')
    
    if success:
        print_success("Frontend dependencies installed")
        return True
    else:
        print_error(f"Failed to install frontend dependencies: {stderr}")
        return False

def create_env_file():
    """Create .env file"""
    print_header("CREATING ENVIRONMENT FILE")
    
    if Path('.env').exists():
        print_warning(".env file already exists")
        return True
    
    env_content = """# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Model Paths
SEGMENTATION_MODEL_PATH=models/saved_models/segmentation/best_dice_model.pth
INFECTION_MODEL_PATH=models/saved_models/infection_detection/best_f1_model.pth
HEALING_STAGE_MODEL_PATH=models/saved_models/healing_stage/best_model.pth

# Database
DATABASE_URL=sqlite:///./wound_tracker.db

# Image Processing
MAX_IMAGE_SIZE=5242880
ALLOWED_EXTENSIONS=jpg,jpeg,png

# Model Configuration
INPUT_SIZE=512
BATCH_SIZE=16
NUM_WORKERS=4

# Frontend URL
FRONTEND_URL=http://localhost:3000
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print_success(".env file created")
    return True

def create_gitignore():
    """Create .gitignore file"""
    print_header("CREATING .gitignore")
    
    if Path('.gitignore').exists():
        print_warning(".gitignore already exists")
        return True
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Data
data/raw/*
data/processed/*
data/augmented/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models
models/saved_models/*.pth
models/saved_models/*.h5
*.pth
*.h5

# Uploads
uploads/*
results/*

# Jupyter
.ipynb_checkpoints
*.ipynb

# Environment
.env
.env.local

# IDEs
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Frontend
frontend/node_modules/
frontend/build/
frontend/.env

# Logs
*.log
logs/

# Database
*.db
*.sqlite
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print_success(".gitignore created")
    return True

def verify_installation():
    """Verify installation"""
    print_header("VERIFYING INSTALLATION")
    
    # Determine python path in venv
    if sys.platform == 'win32':
        python_path = 'venv\\Scripts\\python'
    else:
        python_path = 'venv/bin/python'
    
    # Check key imports
    test_imports = [
        'torch',
        'torchvision',
        'cv2',
        'numpy',
        'fastapi',
        'albumentations'
    ]
    
    all_ok = True
    for module in test_imports:
        success, _, _ = run_command(
            f"{python_path} -c \"import {module}\"",
            check=False
        )
        if success:
            print_success(f"{module}")
        else:
            print_error(f"{module} - NOT FOUND")
            all_ok = False
    
    return all_ok

def create_sample_labels():
    """Create sample label files"""
    print_header("CREATING SAMPLE LABEL FILES")
    
    # Create sample infection labels
    sample_labels = {}
    
    labels_path = Path('data/processed/train/infection_labels.json')
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(labels_path, 'w') as f:
        json.dump(sample_labels, f, indent=2)
    
    print_success("Sample label files created")
    print_info("Add your actual labels to: data/processed/train/infection_labels.json")
    return True

def print_next_steps():
    """Print next steps"""
    print_header("SETUP COMPLETE!")
    
    print(f"{Colors.BOLD}Next Steps:{Colors.ENDC}\n")
    
    print("1. Activate virtual environment:")
    if sys.platform == 'win32':
        print(f"   {Colors.OKCYAN}venv\\Scripts\\activate{Colors.ENDC}")
    else:
        print(f"   {Colors.OKCYAN}source venv/bin/activate{Colors.ENDC}")
    
    print("\n2. Download datasets:")
    print(f"   {Colors.OKCYAN}python preprocessing/download_datasets.py{Colors.ENDC}")
    
    print("\n3. Preprocess data:")
    print(f"   {Colors.OKCYAN}python preprocessing/preprocess_pipeline.py{Colors.ENDC}")
    
    print("\n4. Train models:")
    print(f"   {Colors.OKCYAN}python training/train_segmentation.py{Colors.ENDC}")
    print(f"   {Colors.OKCYAN}python training/train_infection.py{Colors.ENDC}")
    
    print("\n5. Start backend:")
    print(f"   {Colors.OKCYAN}cd backend && uvicorn main:app --reload{Colors.ENDC}")
    
    print("\n6. Start frontend (in new terminal):")
    print(f"   {Colors.OKCYAN}cd frontend && npm start{Colors.ENDC}")
    
    print(f"\n{Colors.OKGREEN}Happy coding! 🚀{Colors.ENDC}\n")

def main():
    """Main setup function"""
    print(f"\n{Colors.BOLD}{Colors.OKBLUE}")
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║                                                           ║")
    print("║        WOUND HEALING TRACKER - SETUP WIZARD              ║")
    print("║                                                           ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}\n")
    
    # Run setup steps
    steps = [
        ("Python Version", check_python_version),
        ("System Dependencies", check_system_dependencies),
        ("Directory Structure", create_directory_structure),
        ("Python Environment", setup_python_environment),
        ("Python Dependencies", install_python_dependencies),
        ("Frontend Setup", setup_frontend),
        ("Environment File", create_env_file),
        (".gitignore", create_gitignore),
        ("Sample Labels", create_sample_labels),
        ("Verification", verify_installation)
    ]
    
    results = {}
    for step_name, step_func in steps:
        try:
            results[step_name] = step_func()
        except Exception as e:
            print_error(f"Error in {step_name}: {str(e)}")
            results[step_name] = False
    
    # Summary
    print_header("SETUP SUMMARY")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for step_name, success in results.items():
        if success:
            print_success(f"{step_name}")
        else:
            print_error(f"{step_name}")
    
    print(f"\n{Colors.BOLD}Success Rate: {success_count}/{total_count}{Colors.ENDC}\n")
    
    if success_count == total_count:
        print_next_steps()
    else:
        print_warning("Setup completed with errors. Please review the output above.")

if __name__ == "__main__":
    main()