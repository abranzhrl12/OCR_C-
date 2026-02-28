#!/bin/bash

# 1. Crear directorio de salida
mkdir -p linux_output

# 2. Construir la imagen Docker
echo "Construyendo imagen Docker para Linux..."
docker build -t ocr-linux-builder -f Dockerfile.linux .

# 3. Ejecutar el contenedor y extraer la librería
echo "Extrayendo libOcrLib.a..."
docker run --rm -v "$(pwd)/linux_output:/output" ocr-linux-builder

echo "--- PROCESO COMPLETADO ---"
echo "La librería estática se encuentra en: linux_output/libOcrLib.a"
