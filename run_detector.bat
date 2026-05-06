@echo off
setlocal

echo ============================================================
echo   Music Tracker: Audio Detection System
echo ============================================================
echo.
echo 1. Index Songs (Upload to Firestore)
echo 2. Start Detection Server (Flask API)
echo 3. Run Local Test Detector (CLI)
echo 4. Exit
echo.
echo ============================================================
set /p choice="Enter choice (1-4): "

if "%choice%"=="1" (
    echo [System] Starting Bulk Indexer...
    python firebase_indexer.py
)
if "%choice%"=="2" (
    echo [System] Starting Flask Backend Server...
    python app.py
)
if "%choice%"=="3" (
    echo [System] Starting Local Fingerprinter CLI...
    python firebase_fingerprinter.py
)
if "%choice%"=="4" (
    echo [System] Exiting.
    exit /b
)

echo.
echo ------------------------------------------------------------
echo Process finished.
pause
