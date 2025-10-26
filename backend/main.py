"""
Updated FastAPI Backend - Uses Simple Segmentation (No Training Required)
Only requires trained infection detection model
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import torch
import cv2
import numpy as np
from pathlib import Path
import uuid
from datetime import datetime
import sys
import os
import traceback


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import simple segmentation
from backend.simple_segmentation import SimpleWoundSegmenter

app = FastAPI(
    title="Wound Healing Tracker API",
    description="API for wound image analysis and tracking",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
infection_model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
segmenter = SimpleWoundSegmenter()

UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

class WoundUploadResponse(BaseModel):
    wound_id: str
    segmentation_mask: str
    area_cm2: float
    perimeter_cm: float
    length_cm: float
    width_cm: float
    infection_risk: float
    infection_level: str
    timestamp: str

class ComparisonRequest(BaseModel):
    wound_id_1: str
    wound_id_2: str

class ComparisonResponse(BaseModel):
    healing_percentage: float
    area_reduction_cm2: float
    healing_velocity: float
    days_between: int
    recommendation: str

@app.on_event("startup")
async def load_models():
    """Load infection detection model on startup"""
    global infection_model
    
    try:
        print("Loading models...")
        
        # Load infection detection model (the one you trained)
        from models.infection_model import InfectionDetectionModel
        infection_model = InfectionDetectionModel(num_classes=1, pretrained=False)
        
        inf_model_path = "models/saved_models/infection_detection/best_f1_model.pth"
        
        if os.path.exists(inf_model_path):
            checkpoint = torch.load(inf_model_path, map_location=device)
            infection_model.load_state_dict(checkpoint['model_state_dict'])
            infection_model.to(device)
            infection_model.eval()
            print("✓ Infection detection model loaded")
        else:
            print("⚠ Infection model not found at:", inf_model_path)
            print("   Using random initialization for demo")
        
        print("✓ Simple segmentation ready (no training required)")
        print(f"✓ Backend ready on {device}")
        
    except Exception as e:
        print(f"✗ Error loading models: {str(e)}")

def preprocess_image(image_bytes, target_size=(512, 512)):
    """Preprocess uploaded image"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    original_size = image.shape[:2]
    image_resized = cv2.resize(image, target_size)
    
    return image, image_resized, original_size

def run_segmentation(image):
    """Run simple segmentation (no model needed!)"""
    mask = segmenter.segment_adaptive(image)
    return mask

def calculate_wound_measurements(mask, pixels_per_cm=50):
    """Calculate wound measurements from binary mask"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return {
            'area_cm2': 0.0,
            'perimeter_cm': 0.0,
            'length_cm': 0.0,
            'width_cm': 0.0
        }
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    area_pixels = cv2.contourArea(largest_contour)
    area_cm2 = area_pixels / (pixels_per_cm ** 2)
    
    perimeter_pixels = cv2.arcLength(largest_contour, True)
    perimeter_cm = perimeter_pixels / pixels_per_cm
    
    x, y, w, h = cv2.boundingRect(largest_contour)
    length_cm = max(w, h) / pixels_per_cm
    width_cm = min(w, h) / pixels_per_cm
    
    return {
        'area_cm2': round(area_cm2, 2),
        'perimeter_cm': round(perimeter_cm, 2),
        'length_cm': round(length_cm, 2),
        'width_cm': round(width_cm, 2)
    }

def run_infection_detection(image):
    """Run infection detection model"""
    # Resize for model
    image_224 = cv2.resize(image, (224, 224))
    
    # Prepare tensor
    image_tensor = torch.from_numpy(image_224).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    image_tensor = (image_tensor - mean) / std
    
    # Predict
    with torch.no_grad():
        output = infection_model(image_tensor)
        probability = torch.sigmoid(output).item()
    
    if probability > 0.7:
        risk_level = "High"
    elif probability > 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"
    
    return probability, risk_level

@app.get("/")
async def root():
    return {
        "message": "Wound Healing Tracker API",
        "version": "2.0.0",
        "status": "running",
        "segmentation": "Simple (no training required)",
        "device": device
    }

@app.get("/health")
async def health_check():
    models_loaded = infection_model is not None
    return {
        "status": "healthy" if models_loaded else "degraded",
        "infection_model_loaded": models_loaded,
        "segmentation": "simple_cv",
        "device": device
    }

@app.post("/upload-wound", response_model=WoundUploadResponse)
async def upload_wound(
    file: UploadFile = File(...),
    patient_id: Optional[str] = Form(None)
):
    """Upload and analyze wound image"""
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_bytes = await file.read()
        wound_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Preprocess
        original_image, image_resized, original_size = preprocess_image(image_bytes)
        
        # Run segmentation (simple, no model needed!)
        mask = run_segmentation(image_resized)
        
        # Calculate measurements
        measurements = calculate_wound_measurements(mask)
        
        # Run infection detection (uses trained model)
        infection_prob, infection_level = run_infection_detection(image_resized)
        
        # Save files
        original_path = UPLOAD_DIR / f"{wound_id}_original.jpg"
        cv2.imwrite(str(original_path), cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
        
        mask_path = UPLOAD_DIR / f"{wound_id}_mask.png"
        cv2.imwrite(str(mask_path), mask)
        
        # Create overlay
        overlay = image_resized.copy()
        mask_colored = np.zeros_like(overlay)
        mask_colored[:, :, 0] = mask  # Red channel
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        overlay_path = RESULTS_DIR / f"{wound_id}_overlay.jpg"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # Save metadata
        import json
        metadata = {
            'wound_id': wound_id,
            'patient_id': patient_id,
            'timestamp': timestamp,
            'measurements': measurements,
            'infection': {
                'probability': float(infection_prob),
                'level': infection_level
            },
            'files': {
                'original': str(original_path),
                'mask': str(mask_path),
                'overlay': str(overlay_path)
            }
        }
        
        metadata_path = UPLOAD_DIR / f"{wound_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return WoundUploadResponse(
            wound_id=wound_id,
            segmentation_mask=f"/results/{wound_id}_overlay.jpg",
            area_cm2=measurements['area_cm2'],
            perimeter_cm=measurements['perimeter_cm'],
            length_cm=measurements['length_cm'],
            width_cm=measurements['width_cm'],
            infection_risk=round(infection_prob, 3),
            infection_level=infection_level,
            timestamp=timestamp
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/compare-wounds", response_model=ComparisonResponse)
async def compare_wounds(request: ComparisonRequest):
    """Compare two wound images"""
    try:
        import json
        
        metadata1_path = UPLOAD_DIR / f"{request.wound_id_1}_metadata.json"
        metadata2_path = UPLOAD_DIR / f"{request.wound_id_2}_metadata.json"
        
        if not metadata1_path.exists() or not metadata2_path.exists():
            raise HTTPException(status_code=404, detail="Wound ID not found")
        
        with open(metadata1_path, 'r') as f:
            metadata1 = json.load(f)
        
        with open(metadata2_path, 'r') as f:
            metadata2 = json.load(f)
        
        area1 = metadata1['measurements']['area_cm2']
        area2 = metadata2['measurements']['area_cm2']
        
        area_reduction = area1 - area2
        healing_percentage = (area_reduction / area1 * 100) if area1 > 0 else 0
        
        date1 = datetime.fromisoformat(metadata1['timestamp'])
        date2 = datetime.fromisoformat(metadata2['timestamp'])
        days_between = abs((date2 - date1).days)
        
        healing_velocity = area_reduction / days_between if days_between > 0 else 0
        
        if healing_percentage > 20:
            recommendation = "Excellent progress! Wound is healing well."
        elif healing_percentage > 10:
            recommendation = "Good progress. Continue current treatment."
        elif healing_percentage > 0:
            recommendation = "Slow progress. Consider consultation with healthcare provider."
        else:
            recommendation = "No improvement or wound has worsened. Seek medical attention."
        
        return ComparisonResponse(
            healing_percentage=round(healing_percentage, 2),
            area_reduction_cm2=round(area_reduction, 2),
            healing_velocity=round(healing_velocity, 3),
            days_between=days_between,
            recommendation=recommendation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing wounds: {str(e)}")

@app.get("/results/{filename}")
async def get_result_file(filename: str):
    """Serve result files"""
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@app.get("/wound/{wound_id}")
async def get_wound_data(wound_id: str):
    """Get all data for a specific wound"""
    import json
    
    metadata_path = UPLOAD_DIR / f"{wound_id}_metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Wound not found")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata


@app.get("/wounds/list")
async def list_all_wounds():
    """List all uploaded wounds"""
    import json
    
    wounds = []
    
    # Find all metadata files
    for metadata_file in UPLOAD_DIR.glob("*_metadata.json"):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                wounds.append({
                    'wound_id': metadata['wound_id'],
                    'timestamp': metadata['timestamp'],
                    'patient_id': metadata.get('patient_id'),
                    'area_cm2': metadata['measurements']['area_cm2'],
                    'infection_level': metadata['infection']['level'],
                    'infection_probability': metadata['infection']['probability']
                })
        except Exception as e:
            print(f"Error reading {metadata_file}: {e}")
            continue
    
    # Sort by timestamp (newest first)
    wounds.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return {'wounds': wounds, 'count': len(wounds)}


@app.get("/wound/{wound_id}/report")
async def generate_report(wound_id: str):
    """Generate PDF report for a wound"""
    try:
        from backend.services.report_generator import generate_wound_report
        
        report_path = generate_wound_report(
            wound_id,
            upload_dir=str(UPLOAD_DIR),
            output_dir="reports"
        )
        
        if report_path and Path(report_path).exists():
            return FileResponse(
                report_path,
                media_type='application/pdf',
                filename=f"wound_report_{wound_id}.pdf"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to generate report")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)