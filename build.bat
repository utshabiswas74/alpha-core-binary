@echo off
title ALPHA CORE V3 - ULTIMATE BUILDER
color 0A

cd /d "%~dp0"

echo ===================================================
echo       ALPHA CORE V3 - CLEAN BUILD SYSTEM
echo ===================================================
echo.
echo [1/4] Cleaning Old Files and Processes...

taskkill /F /IM "Alpha Core.exe" >nul 2>&1
taskkill /F /IM "trainer.exe" >nul 2>&1
taskkill /F /IM "engine.exe" >nul 2>&1
taskkill /F /IM "electron.exe" >nul 2>&1

timeout /t 2 /nobreak >nul

if exist "dist" rmdir /s /q "dist"
if exist "bin" rmdir /s /q "bin"
if exist "build" rmdir /s /q "build"
if not exist "data" mkdir "data"
mkdir "bin"

echo    - Cleanup Complete -
echo.
echo [2/4] Compiling C++ Engines...
g++ -std=c++17 -O3 -fopenmp -static cpp/Train.cpp -o bin/trainer.exe
if %errorlevel% neq 0 goto error

g++ -std=c++17 -O3 -fopenmp -static cpp/Main.cpp -o bin/engine.exe
if %errorlevel% neq 0 goto error
echo    - Engine Ready -
echo.
echo [3/4] Installing Dependencies...
call npm install >nul 2>&1
echo    - Dependencies Installed -
echo.
echo [4/4] Creating Setup File (Wait)...
call npm run dist
if %errorlevel% neq 0 goto error

if not exist "windows" mkdir "windows"
move "dist\*.exe" "windows\" >nul 2>&1
rmdir /s /q "node_modules"
rmdir /s /q "dist"

echo.
echo ===================================================
echo      SUCCESS! Check 'windows' folder.
echo ===================================================
pause
exit

:error
color 0C
echo.
echo [ERROR] BUILD FAILED!
pause
