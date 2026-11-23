@echo off
chcp 65001 >nul
title Iniciando Asistente de Clases

echo [1/3] Iniciando PostgreSQL 18 + pgvector...
start "" /B wsl -d Ubuntu -u root -- bash -c "service postgresql start >/dev/null 2>&1 && tail -f /dev/null"
timeout /t 2 /nobreak >nul

echo [2/3] Levantando servidor FastAPI en http://127.0.0.1:8080 ...
cd /d "%~dp0"
start "CloudLecture-Server" /MIN cmd /c "python -m uvicorn backend.app:app --host 127.0.0.1 --port 8080"

echo [3/3] Esperando a que el servidor cargue el modelo...
:WAIT_LOOP
timeout /t 2 /nobreak >nul
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8080/api/lectures' -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }"
if errorlevel 1 goto WAIT_LOOP

echo Servidor listo. Abriendo navegador...
start "" "http://127.0.0.1:8080"
echo.
echo La plataforma esta corriendo en http://127.0.0.1:8080
echo Usa detener_plataforma.bat cuando quieras apagarla.
timeout /t 3 /nobreak >nul
