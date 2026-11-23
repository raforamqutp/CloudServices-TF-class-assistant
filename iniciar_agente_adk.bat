@echo off
title Google ADK - Servidor Local de Agentes
cd /d "%~dp0"
set PYTHONUTF8=1
echo ===================================================
echo   Iniciando Servidor Web de Google ADK (Gemini)
echo   Acceso local: http://localhost:8000
echo ===================================================
where adk >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    adk web mi_agente --port 8000
) else (
    python -c "from google.adk.cli.cli_tools_click import main; main(['web', 'mi_agente', '--port', '8000'])"
)
pause
