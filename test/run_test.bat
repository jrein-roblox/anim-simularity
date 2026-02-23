@echo off
set "CLI=C:\git\roblox\game-engine2\build\ninja\studio\vs2019\x64\optimized\Client\CLI\app\roblox-cli.exe"
set "REPO=%~dp0"
set "REPO=%REPO:~0,-1%"
"%CLI%" run --run "%REPO%\test_fingerprint.lua" --fs.readwrite "%REPO%" --load.asRobloxScript
exit /b %ERRORLEVEL%
