# reset_run.ps1 — kill stale python processes, wipe DB + evidence, start fresh
# Run from the repo root: .\scripts\reset_run.ps1

$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

Write-Host "--- Stopping all Python processes ---" -ForegroundColor Yellow
Get-Process -Name "python" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "  Killing PID $($_.Id)"
    $_ | Stop-Process -Force
}
Start-Sleep -Seconds 1

Write-Host "--- Deleting database ---" -ForegroundColor Yellow
$dbPath = Join-Path $Root "data\cctv.db"
if (Test-Path $dbPath) {
    Remove-Item $dbPath -Force
    Write-Host "  Deleted $dbPath"
} else {
    Write-Host "  No database found (clean start)"
}

Write-Host "--- Cleaning evidence directory ---" -ForegroundColor Yellow
$evidencePath = Join-Path $Root "data\evidence"
if (Test-Path $evidencePath) {
    Remove-Item $evidencePath -Recurse -Force
    Write-Host "  Cleaned $evidencePath"
}
New-Item -ItemType Directory -Path $evidencePath -Force | Out-Null

Write-Host "--- Starting system ---" -ForegroundColor Green
Write-Host "  URL:   http://localhost:8000"
Write-Host "  Login: admin@local / admin"
Write-Host ""
python scripts\run_all.py

