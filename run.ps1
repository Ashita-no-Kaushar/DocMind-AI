# DocMind AI - Windows Launch Script
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "       Starting DocMind AI System        " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Ensure UTF-8 Console and Python Encoding
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$OutputEncoding = [System.Text.Encoding]::UTF8

# Ensure Git is in PATH if installed in LocalPrograms
if (Test-Path "$env:LOCALAPPDATA\Programs\Git\cmd") {
    $env:PATH = "$env:LOCALAPPDATA\Programs\Git\cmd;$env:LOCALAPPDATA\Programs\Git\bin;" + $env:PATH
}

# Ollama GPU/performance tuning (reduces CPU load and laptop heat)
$env:OLLAMA_FLASH_ATTENTION = "1"          # Faster, lower-power attention on CUDA
$env:OLLAMA_KEEP_ALIVE = "2m"              # Unload models when idle (less heat)
$env:OLLAMA_NUM_PARALLEL = "1"             # One request at a time
$env:OLLAMA_MAX_LOADED_MODELS = "2"        # Keep chat + embedding models resident (no reload churn)
$env:CUDA_VISIBLE_DEVICES = "0"            # Use the discrete NVIDIA GPU, not the iGPU

# Check if Ollama is running
Write-Host "[1/2] Checking Ollama server..." -ForegroundColor Yellow
try {
    $null = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -TimeoutSec 2
    Write-Host "      Ollama server is active and reachable." -ForegroundColor Green
} catch {
    Write-Host "      Starting background Ollama service..." -ForegroundColor Yellow
    if (Test-Path "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe") {
        Start-Process -FilePath "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 2
    } else {
        Write-Host "      Warning: Ollama executable not found in default path. Ensure Ollama is running." -ForegroundColor Red
    }
}

# Start Streamlit application
Write-Host "[2/2] Launching DocMind AI on http://localhost:8501..." -ForegroundColor Yellow
if (Test-Path ".\.venv\Scripts\python.exe") {
    & ".\.venv\Scripts\python.exe" -m streamlit run main.py --server.port=8501 --server.address=127.0.0.1 --server.fileWatcherType=none
} else {
    streamlit run main.py --server.port=8501 --server.address=127.0.0.1 --server.fileWatcherType=none
}
