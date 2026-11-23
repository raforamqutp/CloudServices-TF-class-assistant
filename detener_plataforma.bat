@echo off
chcp 65001 >nul
title Deteniendo Asistente de Clases

echo [1/2] Deteniendo servidor FastAPI (puerto 8080)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8080 " ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)
taskkill /FI "WINDOWTITLE eq CloudLecture-Server*" /F >nul 2>&1

echo [2/2] Deteniendo PostgreSQL en WSL...
wsl -d Ubuntu -u root -- bash -c "service postgresql stop >/dev/null 2>&1; pkill -f 'tail -f /dev/null' >/dev/null 2>&1 || true"

echo.
echo Plataforma detenida correctamente.
timeout /t 2 /nobreak >nul
