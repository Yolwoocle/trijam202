@echo off
:: Handle a switch option
if "%1" == "-c" goto configure
if "%1" == "--configure" goto configure


:build
pyinstaller --clean --log-level WARN SlimyWaterpolo.spec
exit /b

:: Configure
:configure
echo Usage: build.cmd [options]
pyi-makespec --onefile --splash splash.png --log-level WARN --windowed --icon=icon.ico --add-data "data/*;data" --name=SlimyWaterpolo game.py
goto build