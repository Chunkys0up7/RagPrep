@echo off
echo ========================================
echo RAG Document Processing Utility - Demo
echo ========================================
echo.
echo Starting the demo system...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if quick_start.py exists
if not exist "quick_start.py" (
    echo ERROR: quick_start.py not found
    echo Please ensure you're in the correct directory
    pause
    exit /b 1
)

echo Python found. Starting demo...
echo.

REM Run the quick start script
python quick_start.py

echo.
echo Demo completed. Press any key to exit...
pause >nul
