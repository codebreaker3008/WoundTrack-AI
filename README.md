# 🏥 Wound Healing Tracker

AI-powered wound healing progress tracker using deep learning for segmentation, infection detection, and automated measurements.

## 🌟 Features

- **AI-Powered Wound Segmentation**: Automatically detect and segment wound boundaries using U-Net
- **Infection Risk Assessment**: Binary classification with transfer learning (ResNet50/EfficientNet)
- **Automated Measurements**: Calculate wound area, perimeter, length, and width
- **Grad-CAM Explainability**: Visual heatmaps showing which areas indicate infection
- **Progress Tracking**: Compare wounds over time to measure healing velocity
- **RESTful API**: FastAPI backend with async endpoints
- **Modern Web Interface**: React frontend with responsive design

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Training](#model-training)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## 🔧 Prerequisites

### Required Software

- **Python 3.8-3.11** (3.10 recommended)
- **Node.js 16+** and npm
- **Git**
- **CUDA Toolkit** (optional, for GPU acceleration)

### Hardware Requirements

- **Minimum**: 8GB RAM, CPU
- **Recommended**: 16GB RAM, NVIDIA GPU with 6GB+ VRAM

## 📥 Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/wound-healing-tracker.git
cd wound-healing-tracker
```

### 2. Backend Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
python scripts/download_pretrained_models.py
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure Tailwind CSS (if needed)
npx tailwindcss init -p

cd ..
```

### 4. Environment Configuration

Create `.env` file in root directory:

```env
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

SEGMENTATION_MODEL_PATH=models/saved_models/segmentation/best_model.pth
INFECTION_MODEL_PATH=models/saved_models/infection_detection/best_model.pth

DATABASE_URL=sqlite:///./wound_tracker.db
FRONTEND_URL=http://localhost:3000
```
## Images


## 📁 Project Structure
<img width="582" height="528" alt="Screenshot 2025-10-26 204118" src="https://github.com/user-attachments/assets/7c1d9c3d-a74a-4603-9e25-a282319ce061" />
<img width="498" height="263" alt="Screenshot 2025-10-26 204123" src="https://github.com/user-attachments/assets/244d62cd-042e-45d5-b224-474e6260742f" />
<img width="945" height="853" alt="Screenshot 2025-10-26 204140" src="https://github.com/user-attachments/assets/43fdf371-2bef-4490-a13b-bf25f0be64ad" />
<img width="1919" height="1002" alt="Screenshot 2025-10-26 204000" src="https://github.com/user-attachments/assets/7214ba09-19bd-497d-9ca5-1a7042123627" />
<img width="1919" height="999" alt="Screenshot 2025-10-26 204007" src="https://github.com/user-attachments/assets/1751cfc9-96af-4a4f-80c3-4e3f0fa7160e" />
<img width="623" height="616" alt="Screenshot 2025-10-26 204025" src="https://github.com/user-attachments/assets/961018fc-a09b-4a56-a4fa-b46364207f27" />
<img width="950" height="890" alt="Screenshot 2025-10-26 204043" src="https://github.com/user-attachments/assets/9c9008e4-c102-4bdb-a07a-abc6b28c4cfe" />
<img width="570" height="673" alt="Screenshot 2025-10-26 204113" src="https://github.com/user-attachments/assets/3416e8e9-514e-44c5-9127-850134631204" />
<img width="944" height="334" alt="Screenshot 2025-11-04 194802" src="https://github.com/user-attachments/assets/8c9bf3cd-8ef5-47d6-b012-a490335daf88" />
<img width="725" height="654" alt="Screenshot 2025-11-04 195053" src="https://github.com/user-attachments/assets/561c39b3-095c-4a93-bb86-8c2b684a9205" />
<img width="946" height="883" alt="Screenshot 2025-11-04 195107" src="https://github.com/user-attachments/assets/7ace191e-6164-4e53-8c68-84110fdb0cd9" />
<img width="1849" height="507" alt="Screenshot 2025-11-04 194732" src="https://github.com/user-attachments/assets/bd25c61e-8be4-44b4-ab9a-ecd3fd6b1120" />
<img width="953" height="673" alt="Screenshot 2025-11-04 194749" src="https://github.com/user-attachments/assets/9f39a00b-349e-4cd6-a26a-7d7af7963594" />


```
wound-healing-tracker/
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Preprocessed images
│   └── augmented/              # Augmented data
├── models/
│   ├── segmentation_model.py   # U-Net architecture
│   ├── infection_model.py      # Infection classifier
│   ├── gradcam.py             # Grad-CAM implementation
│   └── saved_models/          # Trained checkpoints
├── preprocessing/
│   ├── download_datasets.py    # Dataset downloader
│   ├── preprocess_pipeline.py  # Preprocessing
│   └── augmentation.py        # Data augmentation
├── training/
│   ├── train_segmentation.py  # Train U-Net
│   ├── train_infection.py     # Train classifier
│   └── config.py              # Training configs
├── backend/
│   ├── main.py                # FastAPI application
│   └── services/              # Business logic
├── frontend/
│   └── src/
│       ├── components/        # React components
│       └── services/          # API client
└── requirements.txt
```

## 🚀 Usage

### Start Backend Server

```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Start FastAPI server
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at:
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs

### Start Frontend

```bash
# In a new terminal
cd frontend
npm start
```

Frontend will be available at: http://localhost:3000

### Quick Test

1. Open http://localhost:3000
2. Click "Upload Photo"
3. Select a wound image
4. View AI analysis results

## 📡 API Documentation

### Upload Wound Image

```http
POST /upload-wound
Content-Type: multipart/form-data

Parameters:
  - file: image file (required)
  - patient_id: string (optional)

Response:
{
  "wound_id": "uuid",
  "segmentation_mask": "/results/uuid_overlay.jpg",
  "area_cm2": 4.5,
  "perimeter_cm": 8.2,
  "length_cm": 3.1,
  "width_cm": 2.0,
  "infection_risk": 0.35,
  "infection_level": "Medium",
  "timestamp": "2024-10-25T10:30:00"
}
```

### Compare Wounds

```http
POST /compare-wounds
Content-Type: application/json

Body:
{
  "wound_id_1": "uuid1",
  "wound_id_2": "uuid2"
}

Response:
{
  "healing_percentage": 25.5,
  "area_reduction_cm2": 1.2,
  "healing_velocity": 0.4,
  "days_between": 3,
  "recommendation": "Good progress..."
}
```

### Get Wound Data

```http
GET /wound/{wound_id}

Response:
{
  "wound_id": "uuid",
  "patient_id": "P001",
  "timestamp": "2024-10-25T10:30:00",
  "measurements": {...},
  "infection": {...}
}
```

## 🎓 Model Training

### 1. Download Datasets

```bash
python preprocessing/download_datasets.py
```

Supported datasets:
- DFUC (Diabetic Foot Ulcer Challenge)
- Medetec Wound Database
- AZH Wound Assessment
- Custom datasets

### 2. Preprocess Data

```bash
python preprocessing/preprocess_pipeline.py
```

This will:
- Resize images to 512x512
- Normalize pixel values
- Filter low-quality images
- Create train/val/test splits (70/15/15)

### 3. Data Augmentation

```bash
python preprocessing/augmentation.py
```

Generates augmented versions with:
- Geometric transforms (rotation, flip, elastic deformation)
- Color adjustments (brightness, contrast, saturation)
- Noise and blur simulation
- Skin tone variations

### 4. Train Segmentation Model

```bash
python training/train_segmentation.py
```

Configuration:
- Model: U-Net with skip connections
- Loss: Combined Dice Loss + BCE
- Optimizer: Adam with learning rate scheduling
- Early stopping: patience=15

Training time: ~2-4 hours on GPU

### 5. Train Infection Detection

```bash
python training/train_infection.py
```

Configuration:
- Model: ResNet50 or EfficientNet-B0
- Loss: Weighted BCE or Focal Loss
- Transfer learning: Fine-tune last layers
- Class balancing: Weighted sampling

Training time: ~1-2 hours on GPU

### 6. Evaluate Models

```bash
python training/evaluate.py
```

Metrics:
- Segmentation: IoU, Dice Score, Pixel Accuracy
- Classification: Accuracy, Precision, Recall, F1, ROC-AUC

## 🛠️ Development

### Run Tests

```bash
# Backend tests
pytest tests/

# Frontend tests
cd frontend
npm test
```

### Code Style

```bash
# Python
black .
flake8 .

# JavaScript
npm run lint
```

### Generate Grad-CAM Visualizations

```python
from models.gradcam import generate_infection_heatmap, create_detailed_report
from models.infection_model import InfectionDetectionModel
import torch

# Load model
model = InfectionDetectionModel(num_classes=1)
model.load_state_dict(torch.load('path/to/model.pth'))

# Generate heatmap
result = generate_infection_heatmap(
    model, 
    'path/to/wound_image.jpg',
    device='cuda'
)

# Create report
create_detailed_report(result, 'wound_analysis.png')
```

## 🐛 Troubleshooting

### CUDA Not Detected

```bash
# Reinstall PyTorch with CUDA support
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### OpenCV Import Error

```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python-headless
```

### Frontend Won't Start

```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm start
```

### Model Loading Error

Ensure model files exist:
```bash
ls models/saved_models/segmentation/
ls models/saved_models/infection_detection/
```

If missing, retrain models or download pre-trained weights.

### API CORS Error

Check CORS configuration in `backend/main.py`:
```python
allow_origins=["http://localhost:3000"]
```

## 📊 Performance Benchmarks

### Segmentation Model (U-Net)
- **IoU**: 0.85-0.92
- **Dice Score**: 0.88-0.94
- **Inference Time**: ~50ms (GPU), ~200ms (CPU)

### Infection Detection
- **Accuracy**: 88-93%
- **F1 Score**: 0.87-0.91
- **Inference Time**: ~30ms (GPU), ~150ms (CPU)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## ⚖️ Medical Disclaimer

**IMPORTANT**: This tool is for educational and research purposes only. It is NOT a medical device and should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare professionals for wound care decisions.

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- U-Net architecture based on Ronneberger et al. (2015)
- Transfer learning models from torchvision
- Grad-CAM implementation based on Selvaraju et al. (2017)
- FastAPI and React communities
