Write-Host "Starting helmet detector fine-tuning..."
python scripts\train_helmet_detector.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "Training completed successfully. Restarting server..."
    .\scripts\reset_run.ps1
} else {
    Write-Host "Training failed with exit code $LASTEXITCODE. Not restarting the server."
}
