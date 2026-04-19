@echo off
REM Live Trading Startup Script for Windows
REM ======================================
REM This script starts the trading bot with proper environment setup

echo Starting V5-INSIGNIA Live Trading System
echo ========================================

REM Change to script directory
cd /d "%~dp0.."

REM Activate virtual environment if it exists
if exist "myenv\Scripts\activate.bat" (
    echo Activating Python virtual environment...
    call myenv\Scripts\activate.bat
) else if exist "myenv\bin\activate" (
    echo Activating Python virtual environment...
    call myenv\bin\activate
) else (
    echo WARNING: Virtual environment not found at myenv\Scripts\activate.bat
    echo Please ensure you have created the virtual environment.
)

REM Check if .env file exists
if not exist ".env" (
    echo WARNING: .env file not found!
    echo Please copy .env.example to .env and configure your MT5 credentials:
    echo   copy .env.example .env
    echo   REM Edit .env with your MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
    set /p CONTINUE=Continue anyway? (Y/N):
    if /I not "%CONTINUE%"=="Y" if /I not "%CONTINUE%"=="y" (
        exit /b 1
    )
)

REM Check if config files exist
if not exist "config\config.json" (
    echo ERROR: config\config.json not found!
    exit /b 1
)

REM Start the trading bot
echo.
echo Starting trading bot...
echo Press Ctrl+C to stop
echo ---------------------

REM Default to XAUUSDm if no symbol provided
set "SYMBOL=%~1"
if "%SYMBOL%"=="" set "SYMBOL=XAUUSDm"
set "STRATEGIES=%~2"
if "%STRATEGIES%"=="" set "STRATEGIES=LiquiditySweepBreakout,RangeBounce"

echo Symbol: %SYMBOL%
echo Strategies: %STRATEGIES%
echo ---------------------

python main.py --symbol %SYMBOL% --strategies %STRATEGIES%

REM Deactivate virtual environment if we activated it
if defined VIRTUAL_ENV (
    call deactivate
)