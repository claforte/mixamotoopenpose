@echo off
echo Setting up the virtual environment and installing dependencies...

REM Navigate to the utils folder
cd utils

REM Check if virtual environment already exists
if not exist "venv\Scripts\activate" (
    echo Creating virtual environment in the utils folder...
    python -m venv venv
)

REM Activate the virtual environment
call venv\Scripts\activate

REM Upgrade pip and install requirements
echo Installing dependencies from requirements.txt...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo Setup complete. Virtual environment and dependencies installed.
pause