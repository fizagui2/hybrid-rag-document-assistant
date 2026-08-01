@echo off
setlocal

cd /d "%~dp0"

echo Stopping Hybrid RAG Document Assistant...
docker compose down
if errorlevel 1 (
    echo.
    echo docker compose down failed - see the error above.
    pause
    exit /b 1
)

echo.
echo Stopped. Your indexed documents are preserved for next time.
pause
