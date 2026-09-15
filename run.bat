@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
set EXITCODE=%ERRORLEVEL%
if not "%NOPAUSE%"=="1" pause
exit /b %EXITCODE%
