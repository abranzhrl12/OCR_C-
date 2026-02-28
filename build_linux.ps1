# 1. Preparar carpetas
if (-not (Test-Path "linux_output")) { New-Item -ItemType Directory -Path "linux_output" }

# 2. Construir la imagen de Docker (Optimizado)
Write-Host "Iniciando construccion en Docker para Linux..." -ForegroundColor Cyan
docker build -t ocr-linux-builder -f Dockerfile.linux .

# 3. Ejecutar y Extraer
Write-Host "Extrayendo libOcrLib.a..." -ForegroundColor Green
docker run --rm -v "${PWD}/linux_output:/output" ocr-linux-builder

Write-Host "PROCESO COMPLETADO" -ForegroundColor Green
Write-Host "Archivo generado: linux_output/libOcrLib.a" -ForegroundColor Yellow
