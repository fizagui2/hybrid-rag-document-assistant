@echo off
setlocal

cd /d "%~dp0"

echo Checking Docker is running...
docker info >nul 2>&1
if errorlevel 1 (
    echo.
    echo Docker Desktop does not appear to be running.
    echo Start Docker Desktop, wait for it to finish starting, then run this again.
    echo.
    pause
    exit /b 1
)

echo Starting Hybrid RAG Document Assistant (api + dashboard)...
docker compose up -d
if errorlevel 1 (
    echo.
    echo docker compose up failed - see the error above.
    pause
    exit /b 1
)

echo Waiting for the app to become ready...
set READY=
for /l %%i in (1,1,60) do (
    if not defined READY (
        curl -s -o nul -w "%%{http_code}" http://127.0.0.1:8000/v1/documents 2>nul | findstr "200" >nul
        if not errorlevel 1 (
            curl -s -o nul -w "%%{http_code}" http://127.0.0.1:8501 2>nul | findstr "200" >nul
            if not errorlevel 1 set READY=1
        )
        rem ping as a delay, not "timeout" - timeout errors out when stdin
        rem isn't a real interactive console (e.g. run from some shells/tools)
        if not defined READY ping -n 3 127.0.0.1 >nul
    )
)

if not defined READY (
    echo.
    echo The app didn't become ready in time. Check the logs with:
    echo     docker compose logs
    pause
    exit /b 1
)

echo App is ready. Opening the dashboard in your browser...
start "" http://localhost:8501

echo.
echo Running. To stop it later, run:  docker compose down
echo (This window can be closed - the app keeps running in Docker.)
pause
