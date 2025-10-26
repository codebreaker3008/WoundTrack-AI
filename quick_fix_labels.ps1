# Quick Fix: Create Infection Labels
# Run this after preprocessing

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "   FIXING INFECTION LABELS" -ForegroundColor Cyan
Write-Host "============================================`n" -ForegroundColor Cyan

# Check if processed data exists
if (-not (Test-Path "data\processed\train\images")) {
    Write-Host "❌ Processed data not found!" -ForegroundColor Red
    Write-Host "   Run: python preprocessing/prepare_kaggle_dataset.py" -ForegroundColor Yellow
    exit
}

Write-Host "✅ Found processed data`n" -ForegroundColor Green

# Create infection labels
Write-Host "Creating infection labels (35% infected)...`n" -ForegroundColor Yellow

python preprocessing/create_infection_labels.py

Write-Host "`nLabels created!" -ForegroundColor Green
Write-Host "`nNext step:" -ForegroundColor Cyan
Write-Host '   python training/train_infection.py' -ForegroundColor White
Write-Host ''  # adds an extra newline