@echo off
chcp 65001 >nul
title Panel de Control - Asistente de Clases

:MENU
cls
echo ================================================
echo   ASISTENTE DE CLASES GRABADAS - CONTROL LOCAL
echo ================================================
echo.
echo   [1] Iniciar plataforma (PostgreSQL + Web UI)
echo   [2] Detener plataforma
echo   [3] Abrir navegador (http://127.0.0.1:8080)
echo   [4] Ver estado actual
echo   [5] Salir
echo.
set /p opt=Selecciona una opcion (1-5): 

if "%opt%"=="1" (
    call "%~dp0iniciar_plataforma.bat"
    goto MENU
)
if "%opt%"=="2" (
    call "%~dp0detener_plataforma.bat"
    goto MENU
)
if "%opt%"=="3" (
    start "" "http://127.0.0.1:8080"
    goto MENU
)
if "%opt%"=="4" (
    echo.
    netstat -ano | findstr ":8080 " | findstr "LISTENING" >nul
    if errorlevel 1 (
        echo  - Servidor Web (8080): DETENIDO
    ) else (
        echo  - Servidor Web (8080): ACTIVO
    )
    wsl -d Ubuntu -u root -- service postgresql status >nul 2>&1
    if errorlevel 1 (
        echo  - PostgreSQL (5433):   DETENIDO
    ) else (
        echo  - PostgreSQL (5433):   ACTIVO
    )
    echo.
    pause
    goto MENU
)
if "%opt%"=="5" exit /b 0
goto MENU
