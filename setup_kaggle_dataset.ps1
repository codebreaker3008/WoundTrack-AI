# Complete Kaggle Dataset Setup Script

Write-Host "🏥 Kaggle Wound Dataset Setup" -ForegroundColor Cyan
Write-Host "============================`n" -ForegroundColor Cyan

# Step 1: Install kagglehub
Write-Host "Step 1: Installing kagglehub..." -ForegroundColor Yellow
pip install kagglehub

# Step 2: Check Kaggle credentials
$kaggleDir = "$env:USERPROFILE\.kaggle"
$kaggleJson = "$kaggleDir\kaggle.json"

if (-not (Test-Path $kaggleJson)) {
    Write-Host "`n❌ Kaggle API credentials not found!" -ForegroundColor Red
    Write-Host "📝 Please:" -ForegroundColor Yellow
    Write-Host "   1. Go to https://www.kaggle.com/settings" -ForegroundColor White
    Write-Host "   2. Create API token and download kaggle.json" -ForegroundColor White
    Write-Host "   3. Place it in: $kaggleJson" -ForegroundColor White
    exit
}

Write-Host "✅ Kaggle credentials found" -ForegroundColor Green

# Step 3: Download dataset
Write-Host "`nStep 2: Downloading dataset..." -ForegroundColor Yellow
python preprocessing/download_kaggle_dataset.py

# Step 4: Preprocess
Write-Host "`nStep 3: Preprocessing dataset..." -ForegroundColor Yellow
python preprocessing/prepare_kaggle_dataset.py

Write-Host "`n✅ Setup Complete!" -ForegroundColor Green
Write-Host "`n💡 Next: Train models with:" -ForegroundColor Cyan
Write-Host "   python training/train_infection.py" -ForegroundColor White