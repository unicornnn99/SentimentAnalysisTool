@echo off
echo ==========================================
echo   Starting Sentiment Tool
echo ==========================================

IF NOT EXIST venv (
    echo Creating virtual environment...
    py -3.11 -m venv venv
    call venv\Scripts\activate
    pip install --upgrade pip
    pip install -r requirements.txt --no-cache-dir
) ELSE (
    call venv\Scripts\activate
)

streamlit run app.py

pause
