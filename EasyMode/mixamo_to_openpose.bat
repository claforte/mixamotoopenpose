@echo off
echo Starting conversion process...

REM Check if a file was provided
if "%~1"=="" (
    echo No .dae file provided. Drag and drop a .dae file onto this batch script.
    pause
    exit /b
)

REM Check if the file is a .dae file
if /i not "%~x1"==".dae" (
    echo The provided file is not a .dae file. Please drag a .dae file onto this script.
    pause
    exit /b
)

REM Check if settings.txt exists
if not exist settings.txt (
    echo settings.txt not found. Please ensure it exists in the same directory as this script.
    pause
    exit /b
)

REM Check if virtual environment exists, create it if necessary
if not exist "utils\venv\Scripts\activate" (
    echo Virtual environment not found. Setting up virtual environment...
    pushd utils
    python -m venv venv
    venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    popd
)

REM Activate virtual environment
call utils\venv\Scripts\activate

REM Initialize default settings
set width=512
set height=512
set scale=2.0
set rotation_x=0
set rotation_y=0
set rotation_z=0
set input_fps=30
set max_frames=0
set output_format=GIF

REM Load settings from settings.txt
for /f "tokens=1,2 delims==" %%A in (settings.txt) do (
    set %%A=%%B
)

REM Set the output file path with the format specified in settings
set "output_file=output\%~n1.%output_format%"

REM Run the Python script with settings loaded from settings.txt
python utils\mixamo_to_openpose.py -i "%~1" -o "%output_file%" -ow %width% -oh %height% -os %scale% -rx %rotation_x% -ry %rotation_y% -rz %rotation_z% -ifps %input_fps% -f %max_frames% -of %output_format%

echo Conversion complete. Output saved to %output_file%
pause
