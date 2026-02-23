@echo off
set "CLI=C:\git\roblox\game-engine2\build\ninja\studio\vs2019\x64\optimized\Client\CLI\app\roblox-cli.exe"
set "REPO=%~dp0"
set "REPO=%REPO:~0,-1%"
echo Running group_similar.lua (fingerprint + detailed verification) ...
echo This may take a long time if there are many similar pairs.
"%CLI%" run --run "%REPO%\group_similar.lua" --fs.readwrite "%REPO%" --load.asRobloxScript
exit /b %ERRORLEVEL%
