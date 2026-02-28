cmake_minimum_required(VERSION 3.10)
project(OcrMicroservice LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configuración por Plataforma
if(MSVC)
    # Windows - Optimizado para velocidad y autonomía (Sin DLLs)
    add_compile_options(/arch:AVX2 /O2 /Oi /Ot /GL)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    
    foreach(flag_var 
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE 
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        if(${flag_var} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endif()
    endforeach()
else()
    # Linux / Otros - Optimizado con GCC/Clang
    add_compile_options(-O3 -mavx2 -fPIC)
endif()

# Buscar dependencias
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# de ONNX Runtime (Mas robusto para Linux/Windows)
if(onnxruntime_DIR)
    message(STATUS "Usando onnxruntime_DIR: ${onnxruntime_DIR}")
    include_directories(${onnxruntime_DIR}/include)
    link_directories(${onnxruntime_DIR}/lib)
    
    # Intentar encontrarlo como paquete, si falla, definimos el target nosotros
    find_package(onnxruntime CONFIG QUIET)
    if(NOT onnxruntime_FOUND)
        message(STATUS "onnxruntimeConfig.cmake no encontrado, usando deteccion manual")
        add_library(onnxruntime_lib INTERFACE)
        target_include_directories(onnxruntime_lib INTERFACE ${onnxruntime_DIR}/include)
        if(WIN32)
            target_link_libraries(onnxruntime_lib INTERFACE ${onnxruntime_DIR}/lib/onnxruntime.lib)
        else()
            target_link_libraries(onnxruntime_lib INTERFACE ${onnxruntime_DIR}/lib/libonnxruntime.so)
        endif()
        add_library(onnxruntime::onnxruntime ALIAS onnxruntime_lib)
        set(onnxruntime_FOUND TRUE)
    endif()
else()
    find_package(onnxruntime REQUIRED)
endif()

find_package(nlohmann_json CONFIG REQUIRED)

# Directorios de inclusión
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)

# Archivos fuente del motor OCR
set(OCR_CORE_SOURCES
    src/infrastructure/adapters/OnnxOcrAdapter.cpp
)

# Librerías necesarias
set(OCR_LIBS
    ${OpenCV_LIBS}
    onnxruntime::onnxruntime
    nlohmann_json::nlohmann_json
)

# ===== LIBRERÍA ESTÁTICA (Principal para Rust/Linux) =====
add_library(OcrLib STATIC
    src/ocr_ffi.cpp
    ${OCR_CORE_SOURCES}
)
target_link_libraries(OcrLib PRIVATE ${OCR_LIBS})

# ===== EJECUTABLE (Para pruebas locales) =====
add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE OcrLib ${OCR_LIBS})

