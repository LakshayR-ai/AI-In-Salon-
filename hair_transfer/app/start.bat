@echo off
echo Installing app dependencies...
pip install -r "%~dp0requirements.txt" -q

echo Starting AI Salon...
cd /d "%~dp0.."
uvicorn app.backend:app --host 0.0.0.0 --port 8000 --reload

pause
