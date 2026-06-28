@echo off
setlocal EnableDelayedExpansion
title Agnel Match Predicting Algorithm (AMPA)

:MENU
cls
echo ==========================================================
echo        AGNEL MATCH PREDICTING ALGORITHM (AMPA)
echo ==========================================================
echo.
echo 1. Run Prediction Engine (AMPE)
echo 2. Launch Interface (AMPI)
echo 3. View Saves
echo 4. Delete a Save
echo 5. Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto RUN_ENGINE
if "%choice%"=="2" goto LAUNCH_UI
if "%choice%"=="3" goto VIEW_SAVES
if "%choice%"=="4" goto DELETE_SAVE
if "%choice%"=="5" goto EXIT
goto MENU

:RUN_ENGINE
cls
echo ==========================================================
echo        INITIALIZING AMPE
echo ==========================================================
echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10+ from python.org
    pause
    goto MENU
)

echo Checking/Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Created new virtual environment.
)

echo Activating environment and checking dependencies...
call venv\Scripts\activate.bat
pip install -q pandas requests scipy numpy
echo Dependencies ready.
echo.
echo Running AMPE...
python engine\main.py --quiet
echo.
echo Press any key to return to menu...
pause >nul
goto MENU

:LAUNCH_UI
cls
echo Launching Agnel Match Prediction Interface (AMPI)...
start ui\index.html
goto MENU

:VIEW_SAVES
cls
echo ==========================================================
echo        SAVED PREDICTIONS (outputs folder)
echo ==========================================================
python engine\ampe_helper.py view
echo.
echo Press any key to return to menu...
pause >nul
goto MENU

:DELETE_SAVE
cls
echo ==========================================================
echo        DELETE A SAVE
echo ==========================================================
python engine\ampe_helper.py view
echo.
set /p del_idx="Enter the number of the save to delete (or leave blank to cancel): "
if "%del_idx%"=="" goto MENU
python engine\ampe_helper.py delete %del_idx%
echo.
pause
goto MENU

:EXIT
exit
