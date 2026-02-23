@echo off
REM Run anim_sim_pipeline.lua with mode and options passed after -- (command line args).
REM Usage: run_pipeline.bat download | fingerprint | group [extra options]
REM Example: run_pipeline.bat group --threshold 0.99

set "MODE=%~1"
if "%MODE%"=="" (
  echo Usage: run_pipeline.bat download ^| fingerprint ^| group [--opt value ...]
  exit /b 1
)

set "CLI=C:\git\roblox\game-engine2\build\ninja\studio\vs2019\x64\optimized\Client\CLI\app\roblox-cli.exe"
set "REPO=%~dp0"
set "REPO=%REPO:~0,-1%"

set "ARGS=--mode %MODE% --base ."
if "%MODE%"=="download" set "ARGS=%ARGS% --input animations.csv"
if "%MODE%"=="group" set "ARGS=%ARGS% --input fingerprints.csv"

REM Pass any extra args (e.g. run_pipeline.bat group --threshold 0.98 --skipDetailVerify)
shift
:loop
if "%~1"=="" goto run
set "ARGS=%ARGS% %1"
shift
goto loop
:run

echo Running pipeline: %MODE%
"%CLI%" run --run "%REPO%\anim_sim_pipeline.lua" --fs.readwrite "%REPO%" --load.asRobloxScript -- %ARGS%
exit /b %ERRORLEVEL%
