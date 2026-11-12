@echo off
rem ============================================================
rem  Sign2Speech  -  move this project OUT of OneDrive
rem
rem  WHY:  Your project is 8.9 GB but your free OneDrive only
rem        has 5 GB.  OneDrive can't sync it, so every file
rem        shows a red X.  Moving it out fixes the red X.
rem
rem  Nothing is deleted.  The folder is moved to:
rem      C:\Users\hrush\Projects\AiSignToSpeech
rem
rem  HOW TO USE:
rem      1) CLOSE Trae / VS Code first (it locks the folder!).
rem      2) Open File Explorer, go to your Desktop.
rem      3) Double-click this file: MoveOutOfOneDrive.bat
rem ============================================================
setlocal enableextensions
title Sign2Speech - move out of OneDrive

set "SRC=C:\Users\hrush\OneDrive\Desktop\AiSignToSpeech"
set "DST=C:\Users\hrush\Projects\AiSignToSpeech"
set "ME=%TEMP%\s2s_move.bat"
set "ODEXE=C:\Program Files\Microsoft OneDrive\OneDrive.exe"

rem Relaunch a copy from TEMP so this folder isn't locked while moving.
if /i not "%~1"=="--run" (
  copy /y "%~f0" "%ME%" >nul
  start "s2smove" cmd /c "%ME% --run"
  exit /b
)

cd /d "%TEMP%"
timeout /t 1 /nobreak >nul

echo.
echo ============================================================
echo   Sign2Speech - moving OUT of OneDrive
echo ============================================================
echo   From: %SRC%
echo   To:   %DST%
echo.

if not exist "%SRC%" (
  echo   The project is NOT in the old location - it was probably
  echo   already moved.  Opening C:\Users\hrush\Projects ...
  start "" explorer "C:\Users\hrush\Projects"
  echo.
  pause
  exit /b 0
)

rem Warn if Trae / VS Code is still running (folder will be locked).
tasklist /fi "imagename eq Trae.exe" 2>nul | find /i "Trae.exe" >nul
if not errorlevel 1 (
  echo   *** WARNING: Trae is still running! ***
  echo   Please close Trae, then run this script again.
  echo   (Trae keeps the project folder locked, so the move will fail.)
  echo.
  pause
  exit /b 1
)
tasklist /fi "imagename eq Code.exe" 2>nul | find /i "Code.exe" >nul
if not errorlevel 1 (
  echo   *** WARNING: VS Code is still running! ***
  echo   Please close VS Code, then run this script again.
  echo.
  pause
  exit /b 1
)

if not exist "C:\Users\hrush\Projects" mkdir "C:\Users\hrush\Projects" >nul 2>&1

echo   [1/5] Stopping OneDrive (releases the files)...
taskkill /f /im OneDrive.exe >nul 2>&1

echo   [2/5] Stopping any running Python / app server...
taskkill /f /im python.exe >nul 2>&1
taskkill /f /im pythonw.exe >nul 2>&1
timeout /t 2 /nobreak >nul

echo   [3/5] Moving the folder (same drive - should be quick)...
move "%SRC%" "%DST%" >nul 2>&1
if errorlevel 1 (
  echo       move failed, trying robocopy as backup...
  robocopy "%SRC%" "%DST%" /E /MOVE /R:1 /W:1 /NFL /NDL /NJH /NJS >nul 2>&1
  if errorlevel 8 (
    echo.
    echo   *** MOVE FAILED ***
    echo.
    echo   The folder is still locked by another program.
    echo   Make sure Trae / VS Code / File Explorer are closed,
    echo   then double-click this file again.
    echo.
    pause
    exit /b 1
  )
)

echo   [4/5] Restarting OneDrive...
if exist "%ODEXE%" (
  start "" "%ODEXE%"
) else if exist "%LOCALAPPDATA%\Microsoft\OneDrive\OneDrive.exe" (
  start "" "%LOCALAPPDATA%\Microsoft\OneDrive\OneDrive.exe"
)

echo   [5/5] Done!
echo.
echo ============================================================
echo   SUCCESS!  The red X symbols will disappear.
echo ============================================================
echo.
echo   Your project is now at:
echo       %DST%
echo.
echo   NEXT STEPS:
echo     1) Open Trae
echo     2) File  -^>  Open Folder  -^>  C:\Users\hrush\Projects\AiSignToSpeech
echo     3) In the terminal run:  .venv\Scripts\python.exe app.py
echo.
start "" explorer "%DST%"
pause
