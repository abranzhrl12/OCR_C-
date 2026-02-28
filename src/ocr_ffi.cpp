#include "ocr_ffi.h"
#include "infrastructure/adapters/OnnxOcrAdapter.hpp"
#include "application/ProcessImageUseCase.hpp"
#include "domain/services/LayoutService.hpp"
#include "domain/builders/TextBuilderFactory.hpp"
#include <string>
#include <cstring>
#include <memory>
#include <iostream>

using namespace ocr::infrastructure;
using namespace ocr::application;
using namespace ocr::domain;

// Estructura para envolver el UseCase y el Motor, ya que la FFI usa punteros opacos (void*)
struct OcrContext {
    std::shared_ptr<OnnxOcrAdapter> adapter;
    std::unique_ptr<ProcessImageUseCase> useCase;
};

// --- Helpers Internos ---

static char* duplicateString(const std::string& str) {
    char* result = new char[str.size() + 1];
    std::memcpy(result, str.c_str(), str.size() + 1);
    return result;
}

/**
 * @brief Realiza el formateo bajo demanda (Lazy Formatting) para no penalizar el rendimiento.
 */
static const char* formatResult(const std::vector<TextBlock>& blocks, TextFormat format) {
    auto builder = TextBuilderFactory::create(format);
    if (!builder) return nullptr;

    // Para formatos que requieren estructura de líneas
    if (format == TextFormat::RAW_JSON) {
        return duplicateString(LayoutService::build(blocks, {}, *builder));
    }

    auto lines = LayoutService::groupIntoLines(blocks);
    return duplicateString(LayoutService::build(blocks, lines, *builder));
}

// --- API de C ---

extern "C" {

OCR_API void* ocr_init(const char* det_model_path,
                       const char* rec_model_path,
                       const char* dict_path) {
    try {
        OcrConfig config;
        config.detModelPath = det_model_path;
        config.recModelPath = rec_model_path;
        config.dictPath = dict_path;
        
        auto context = new OcrContext();
        context->adapter = std::make_shared<OnnxOcrAdapter>(config);
        context->useCase = std::make_unique<ProcessImageUseCase>(context->adapter);
        
        return static_cast<void*>(context);
    } catch (const std::exception& e) {
        std::cerr << "[OCR_FFI] Error inicializando: " << e.what() << std::endl;
        return nullptr;
    }
}

// Helper para loguear errores de Result
template<typename T>
static bool checkResult(const Result<T>& res, const char* context) {
    if (!res) {
        std::cerr << "[OCR_FFI] Error en " << context << ": " << res.error().message << std::endl;
        return false;
    }
    return true;
}


OCR_API const char* ocr_process(void* handle, const char* image_path, int mode) {
    if (!handle || !image_path) return nullptr;
    try {
        auto* ctx = static_cast<OcrContext*>(handle);
        auto result = ctx->useCase->execute(image_path);
        if (!checkResult(result, "ocr_process")) return nullptr;

        TextFormat fmt = (mode == OCR_MODE_PLAIN) ? TextFormat::PLAIN : TextFormat::LAYOUT;
        return formatResult(result.value(), fmt);
    } catch (...) { return nullptr; }
}

OCR_API const char* ocr_process_buffer(void* handle, const unsigned char* buffer, int size, int mode) {
    if (!handle || !buffer || size <= 0) return nullptr;
    try {
        auto* ctx = static_cast<OcrContext*>(handle);
        auto result = ctx->useCase->execute(buffer, (size_t)size);
        if (!checkResult(result, "ocr_process_buffer")) return nullptr;

        TextFormat fmt = (mode == OCR_MODE_PLAIN) ? TextFormat::PLAIN : TextFormat::LAYOUT;
        return formatResult(result.value(), fmt);
    } catch (...) { return nullptr; }
}

OCR_API const char* ocr_process_json(void* handle, const char* image_path) {
    if (!handle || !image_path) return nullptr;
    try {
        auto* ctx = static_cast<OcrContext*>(handle);
        auto result = ctx->useCase->execute(image_path);
        if (!checkResult(result, "ocr_process_json")) return nullptr;
        return formatResult(result.value(), TextFormat::JSON);
    } catch (...) { return nullptr; }
}

OCR_API const char* ocr_process_buffer_json(void* handle, const unsigned char* buffer, int size) {
    if (!handle || !buffer || size <= 0) return nullptr;
    try {
        auto* ctx = static_cast<OcrContext*>(handle);
        auto result = ctx->useCase->execute(buffer, (size_t)size);
        if (!checkResult(result, "ocr_process_buffer_json")) return nullptr;
        return formatResult(result.value(), TextFormat::JSON);
    } catch (...) { return nullptr; }
}

OCR_API const char* ocr_process_raw(void* handle, const char* image_path) {
    if (!handle || !image_path) return nullptr;
    try {
        auto* ctx = static_cast<OcrContext*>(handle);
        auto result = ctx->useCase->execute(image_path);
        if (!checkResult(result, "ocr_process_raw")) return nullptr;
        return formatResult(result.value(), TextFormat::RAW_JSON);
    } catch (...) { return nullptr; }
}

OCR_API const char* ocr_process_buffer_raw(void* handle, const unsigned char* buffer, int size) {
    if (!handle || !buffer || size <= 0) return nullptr;
    try {
        auto* ctx = static_cast<OcrContext*>(handle);
        auto result = ctx->useCase->execute(buffer, (size_t)size);
        if (!checkResult(result, "ocr_process_buffer_raw")) return nullptr;
        return formatResult(result.value(), TextFormat::RAW_JSON);
    } catch (...) { return nullptr; }
}

OCR_API const char* ocr_process_pixels_raw(void* handle, const unsigned char* pixels, int width, int height, int channels) {
    if (!handle || !pixels || width <= 0 || height <= 0) return nullptr;
    try {
        auto* ctx = static_cast<OcrContext*>(handle);
        int type = (channels == 3) ? CV_8UC3 : CV_8UC1;
        cv::Mat pixelsMat(height, width, type, const_cast<unsigned char*>(pixels));
        
        auto result = ctx->useCase->execute(pixelsMat);
        if (!checkResult(result, "ocr_process_pixels_raw")) return nullptr;
        
        return formatResult(result.value(), TextFormat::RAW_JSON);
    } catch (...) { return nullptr; }
}


OCR_API void ocr_free_string(const char* str) {
    delete[] str;
}

OCR_API void ocr_destroy(void* handle) {
    if (handle) {
        delete static_cast<OcrContext*>(handle);
    }
}

} // extern "C"

