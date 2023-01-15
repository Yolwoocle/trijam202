@echo off
echo Usage: build.cmd [options]
:: Handle a switch option
if "%1" == "-c" goto configure
if "%1" == "--configure" goto configure


:build
pyinstaller --clean --log-level WARN Trijam202.spec
exit /b

:: Configure
:configure
pyi-makespec --onefile --splash data/splash.png --log-level WARN --windowed --icon=icon.ico --add-data "data/*;data" --add-data "assets/art/*;assets/art" --name=Trijam202 game.py
goto build