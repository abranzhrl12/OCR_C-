#ifndef INFRASTRUCTURE_ONNX_SESSION_HELPER_HPP
#define INFRASTRUCTURE_ONNX_SESSION_HELPER_HPP

#include <onnxruntime_cxx_api.h>
#include <string>
#include <thread>
#include <algorithm>
#include <cmath>
#include "PlatformUtils.hpp"

namespace ocr::infrastructure {

/**
 * @brief Helper para configurar y crear sesiones de ONNX Runtime de manera consistente.
 */
class OnnxSessionHelper {
public:
    /**
     * @brief Crea una sesión optimizada.
     */
    static Ort::Session createSession(Ort::Env& env, const std::string& modelPath) {
        Ort::SessionOptions options;
        
        // Optimización por defecto
        options.SetLogSeverityLevel(4); // Solo fatal
        unsigned int cpuCores = std::thread::hardware_concurrency();
        options.SetIntraOpNumThreads(std::max(2, (int)std::sqrt((float)cpuCores)));
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef _WIN32
        std::wstring wPath = PathUtils::toOnnxPath(modelPath);
        return Ort::Session(env, wPath.c_str(), options);
#else
        return Ort::Session(env, modelPath.c_str(), options);
#endif
    }
};

} // namespace ocr::infrastructure

#endif
