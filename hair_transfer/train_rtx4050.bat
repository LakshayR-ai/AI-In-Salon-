@echo off
echo ============================================
echo  AI Salon - RTX 4050 Optimized Training
echo ============================================

cd /d "%~dp0"

REM Memory optimizations for 6GB VRAM
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set CUDA_LAUNCH_BLOCKING=0

REM Stage 1 - Encoder
echo.
echo [1/4] Training Encoder...
python training/train_encoder.py
if errorlevel 1 goto error

REM Stage 2 - Shape Module
echo.
echo [2/4] Training Shape Module...
python training/train_shape_module.py
if errorlevel 1 goto error

REM Stage 3 - Color Module
echo.
echo [3/4] Training Color Module...
python training/train_color_module.py
if errorlevel 1 goto error

REM Stage 4 - Refinement
echo.
echo [4/4] Training Refinement...
python training/train_refinement.py
if errorlevel 1 goto error

echo.
echo ============================================
echo  Training Complete!
echo  Run: uvicorn app.backend:app --port 8000
echo ============================================
goto end

:error
echo Training failed at current stage.
pause

:end
pause
