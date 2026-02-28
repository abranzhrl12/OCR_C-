#ifndef INFRASTRUCTURE_PLATFORM_UTILS_HPP
#define INFRASTRUCTURE_PLATFORM_UTILS_HPP

#include <string>

namespace ocr::infrastructure {

/**
 * @brief Clase RAII (Resource Acquisition Is Initialization) para silenciar la salida de error (stderr).
 * 
 * Al instanciarse, redirige la salida estándar de error hacia el dispositivo nulo (NUL/dev/null).
 * Al ser destruida, restaura automáticamente la salida original.
 */
class ScopedLogSilencer {
public:
    /**
     * @brief Constructor que activa el silenciamiento.
     * @param enable Si es falso, no realiza ninguna acción.
     */
    explicit ScopedLogSilencer(bool enable = true);

    /**
     * @brief Destructor que restaura la salida original.
     */
    ~ScopedLogSilencer();

    // No permitir copia para evitar problemas con descriptores de archivos
    ScopedLogSilencer(const ScopedLogSilencer&) = delete;
    ScopedLogSilencer& operator=(const ScopedLogSilencer&) = delete;

private:
    int oldStderr_;
    bool isSilenced_;
};

/**
 * @brief Utilidades para manejo de rutas de archivos en distintas plataformas.
 * 
 * ONNX Runtime en Windows requiere rutas en formato wchar_t (UTF-16).
 */
class PathUtils {
public:
    /**
     * @brief Convierte una ruta en string (UTF-8) a formato compatible con ONNX en Windows.
     */
#ifdef _WIN32
    static std::wstring toOnnxPath(const std::string& path);
#else
    static const char* toOnnxPath(const std::string& path);
#endif
};

} // namespace ocr::infrastructure

#endif // INFRASTRUCTURE_PLATFORM_UTILS_HPP
